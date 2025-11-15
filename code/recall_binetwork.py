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

parser = argparse.ArgumentParser(description='binetwork 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'binetwork 召回，mode: {mode}')


def cal_sim(df):
    """
    计算物品相似度矩阵，基于二部图(BiNetwork)的物品协同过滤算法
    
    参数:
        df (DataFrame): 包含用户点击历史的数据，至少包含user_id和click_article_id列
    
    返回:
        tuple: (sim_dict, user_item_dict)
            sim_dict: 物品相似度字典，格式为{item_id: {related_item_id: similarity_score}}
            user_item_dict: 用户历史交互物品字典，格式为{user_id: [item_id1, item_id2, ...]}
    """    
    user_item_ = df.groupby('user_id')['click_article_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(list).reset_index()
    item_user_dict = dict(zip(item_user_['click_article_id'], item_user_['user_id']))

    sim_dict = {}

    for item, users in tqdm(item_user_dict.items()):
        sim_dict.setdefault(item, {})

        for user in users:
            tmp_len = len(user_item_dict[user])
            for relate_item in user_item_dict[user]:
                sim_dict[item].setdefault(relate_item, 0)
                sim_dict[item][relate_item] += 1 / (math.log(len(users)+1) * math.log(tmp_len+1))

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    """
    基于二部图(BiNetwork)的物品协同过滤召回函数
    
    参数:
        df_query (DataFrame): 包含用户ID和目标文章ID的查询数据
        item_sim (dict): 物品相似度字典，由cal_sim函数计算得出
        user_item_dict (dict): 用户历史交互物品字典，由cal_sim函数计算得出
        worker_id (int): 工作进程ID，用于保存中间结果
    
    功能:
        1. 对于每个用户，获取其最近交互的物品
        2. 基于物品相似度，找到与用户最近交互物品最相似的物品
        3. 计算推荐分数并排序
        4. 生成召回结果并保存到文件
    """    
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:1]

        for _, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
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
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/binetwork', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/binetwork/{worker_id}.pkl')


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_path = os.path.join(project_root, 'user_data')
    data_path = os.path.join(user_data_path, 'data')
    sim_path = os.path.join(user_data_path, 'sim')
    
    if mode == 'valid':
        df_click = pd.read_pickle(os.path.join(data_path, 'offline/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'offline/query.pkl'))

        os.makedirs(os.path.join(sim_path, 'offline'), exist_ok=True)
        sim_pkl_file = os.path.join(sim_path, 'offline/binetwork_sim.pkl')
    else:
        df_click = pd.read_pickle(os.path.join(data_path, 'online/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'online/query.pkl'))

        os.makedirs(os.path.join(sim_path, 'online'), exist_ok=True)
        sim_pkl_file = os.path.join(sim_path, 'online/binetwork_sim.pkl')

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

    tmp_path = os.path.join(user_data_path, 'tmp/binetwork')
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

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk(os.path.join(user_data_path, 'tmp/binetwork')):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], ignore_index=True)

    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

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
        log.debug(f'accuracy: {accuracy}')

    if mode == 'valid':
        df_data.to_pickle(os.path.join(data_path, 'offline/recall_binetwork.pkl'))
    else:
        df_data.to_pickle(os.path.join(data_path, 'online/recall_binetwork.pkl'))