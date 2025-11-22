import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# Command line arguments
parser = argparse.ArgumentParser(description='Recall merge')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# Initialize logger
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'Recall merge: {mode}')


def mms(df):
    """
    对每个用户的相似度得分进行最小-最大标准化处理
    
    该函数通过计算每个用户相似度得分的最大值和最小值，
    对相似度得分进行归一化处理，避免不同用户间得分差异过大。
    
    参数:
        df (DataFrame): 包含用户ID和相似度得分的数据框
        
    返回:
        list: 归一化后的相似度得分列表
    """
    user_score_max = {}
    user_score_min = {}

    # Get the maximum and minimum similarity scores for each user
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    """
    计算两种召回方法之间的相似度，即共同召回的文章比例
    
    该函数用于衡量不同召回方法之间的重叠程度，通过计算共同召回的文章
    数量占总召回文章数量的比例来评估召回方法间的相似性。
    
    参数:
        df1_ (DataFrame): 第一种召回方法的结果数据框
        df2_ (DataFrame): 第二种召回方法的结果数据框
        
    返回:
        float: 两种召回方法的相似度值
    """
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    recall_methods = ['itemcf', 'w2v', 'binetwork']

    weights = {'itemcf': 1, 'binetwork': 1, 'w2v': 0.1}
    recall_list = []
    recall_dict = {}
   
    for recall_method in recall_methods:
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # Calculate similarity
    
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'Recall similarity {recall_method1}-{recall_method2}: {score}')

    # Merge recall results
    recall_final = pd.concat(recall_list, sort=False)
    recall_score = recall_final[['user_id', 'article_id','sim_score']].groupby(['user_id', 'article_id'])['sim_score'].sum().reset_index()
    
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    recall_final = recall_final.merge(recall_score, how='left')

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # Remove training set users without positive samples
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # Calculate metrics
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50, accuracy = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)

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

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"Average number of recalls per user: {df['cnt'].mean()}")

    log.debug(
        f"Label distribution: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # Save to local file
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')