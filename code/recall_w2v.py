import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

# 添加项目根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    """
    使用Word2Vec算法训练文章向量表示
    
    该函数通过对用户历史点击序列进行建模，训练出文章的向量表示。
    这些向量可以用于计算文章之间的相似度，从而实现基于内容的召回。
    
    参数:
        df_ (DataFrame): 输入数据框，包含用户点击行为数据
        f1 (str): 分组字段名，通常是'user_id'，用于构建用户点击序列
        f2 (str): 序列字段名，通常是'click_article_id'，表示用户点击的文章ID
        model_path (str): Word2Vec模型保存路径
    
    返回:
        dict: 文章向量映射字典，key为文章ID，value为对应的向量表示
    """
    df = df_.copy()
    
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         vector_size=256,
                         window=3,
                         min_count=1,
                         sg=1,
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10,
                         epochs=1)
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model.wv.key_to_index:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):
    """
    基于Word2Vec向量相似度进行召回
    
    该函数使用训练好的文章向量，通过计算向量相似度找到与用户历史
    点击文章相似的候选文章，形成推荐列表。
    
    参数:
        df_query (DataFrame): 查询数据框，包含用户ID和待预测文章ID
        article_vec_map (dict): 文章向量映射字典
        article_index (AnnoyIndex): 使用Annoy构建的文章向量索引，用于快速近似最近邻搜索
        user_item_dict (dict): 用户历史点击文章字典，key为用户ID，value为点击过的文章列表
        worker_id (int): 工作进程ID，用于保存结果文件命名
    """
    data_list = []


    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-2:]

        for item in interacted_items:
            article_vec = article_vec_map[item]

            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 200, include_distances=True)
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
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

    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_path = os.path.join(project_root, 'user_data')
    data_path = os.path.join(user_data_path, 'data')
    model_path = os.path.join(user_data_path, 'model')
    
    if mode == 'valid':
        df_click = pd.read_pickle(os.path.join(data_path, 'offline/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'offline/query.pkl'))

        os.makedirs(os.path.join(data_path, 'offline'), exist_ok=True)
        os.makedirs(os.path.join(model_path, 'offline'), exist_ok=True)

        w2v_file = os.path.join(data_path, 'offline/article_w2v.pkl')
        model_path = os.path.join(model_path, 'offline')
    else:
        df_click = pd.read_pickle(os.path.join(data_path, 'online/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'online/query.pkl'))

        os.makedirs(os.path.join(data_path, 'online'), exist_ok=True)
        os.makedirs(os.path.join(model_path, 'online'), exist_ok=True)

        w2v_file = os.path.join(data_path, 'online/article_w2v.pkl')
        model_path = os.path.join(model_path, 'online')

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    tmp_path = os.path.join(user_data_path, 'tmp/w2v')
    os.makedirs(tmp_path, exist_ok=True)
    for path, _, file_list in os.walk(tmp_path):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_list = []
    for path, _, file_list in os.walk(os.path.join(user_data_path, 'tmp/w2v')):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_list.append(df_temp)
    
    df_data = pd.concat(df_list, sort=False)

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
    if mode == 'valid':
        df_data.to_pickle(os.path.join(data_path, 'offline/recall_w2v.pkl'))
    else:
        df_data.to_pickle(os.path.join(data_path, 'online/recall_w2v.pkl'))