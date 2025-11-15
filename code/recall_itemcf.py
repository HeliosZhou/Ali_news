import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def cal_sim(df):
    """
    计算物品间的相似度
    
    参数:
    df: 包含用户点击数据的DataFrame，至少包含'user_id'和'click_article_id'两列
    
    返回:
    sim_dict: 物品相似度字典，key为物品ID，value为与其它物品的相似度
    user_item_dict: 用户-物品倒排表，key为用户ID，value为该用户点击过的物品列表
    """
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))
    
    item_cnt = defaultdict(int)
    sim_dict = {}

    for _, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    """
    基于物品相似度进行召回
    
    参数:
    df_query: 查询数据，包含用户ID和物品ID
    item_sim: 物品相似度字典
    user_item_dict: 用户-物品倒排表
    worker_id: 多线程任务ID
    """
    data_list = []
    for user_id, item_id in tqdm(df_query.values):
        rank = {}
        if user_id not in user_item_dict:
            continue
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items):
            if item not in item_sim:
                continue
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            if item_id in item_ids:
                df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)
    
    if data_list:
        df_data = pd.concat(data_list, sort=False)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_path = os.path.join(project_root, 'user_data')
    data_path = os.path.join(user_data_path, 'data')
    
    if mode == 'valid':
        df_click = pd.read_pickle(os.path.join(data_path, 'offline/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'offline/query.pkl'))

        os.makedirs(os.path.join(user_data_path, 'sim/offline'), exist_ok=True)
        sim_pkl_file = os.path.join(user_data_path, 'sim/offline/itemcf_sim.pkl')
    else:
        df_click = pd.read_pickle(os.path.join(data_path, 'online/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'online/query.pkl'))

        os.makedirs(os.path.join(user_data_path, 'sim/online'), exist_ok=True)
        sim_pkl_file = os.path.join(user_data_path, 'sim/online/itemcf_sim.pkl')

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    tmp_path = os.path.join(user_data_path, 'tmp/itemcf')
    os.makedirs(tmp_path, exist_ok=True)
    for path, _, file_list in os.walk(tmp_path):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    data_list = []
    for path, _, file_list in os.walk(tmp_path):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            data_list.append(df_temp)
    
    if data_list:
        df_data = pd.concat(data_list, sort=False).reset_index(drop=True)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')
    log.debug(f'df_data shape: {df_data.shape}')

    if mode == 'valid':
        log.info(f'计算召回指标')
        
        log.debug(f"召回结果中的用户数: {df_data['user_id'].nunique()}")
        log.debug(f"召回结果总数: {len(df_data)}")
        log.debug(f"有标签的数据数: {len(df_data[df_data['label'].notnull()])}")
        
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        log.debug(f"查询数据中的用户数: {total}")

        if len(df_data[df_data['label'].notnull()]) > 0:
            hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50, accuracy = evaluate(
                df_data[df_data['label'].notnull()], total)

            log.debug(
                f'itemcf: \n'
                f'accuracy: \t{accuracy:.4f}\n'
                f'hitrate_5: \t{hitrate_5:.4f}\n'
                f'mrr_5: \t{mrr_5:.4f}\n'
                f'hitrate_10: \t{hitrate_10:.4f}\n'
                f'mrr_10: \t{mrr_10:.4f}\n'
                f'hitrate_20: \t{hitrate_20:.4f}\n'
                f'mrr_20: \t{mrr_20:.4f}\n'
                f'hitrate_40: \t{hitrate_40:.4f}\n'
                f'mrr_40: \t{mrr_40:.4f}\n'
                f'hitrate_50: \t{hitrate_50:.4f}\n'
                f'mrr_50: \t{mrr_50:.4f}'
            )
        else:
            log.debug("没有找到带标签的数据，无法计算评估指标")

    if mode == 'valid':
        df_data.to_pickle(os.path.join(data_path, 'offline/recall_itemcf.pkl'))
    else:
        df_data.to_pickle(os.path.join(data_path, 'online/recall_itemcf.pkl'))