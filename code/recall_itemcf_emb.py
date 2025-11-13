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
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 添加项目根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger, evaluate

# 添加PyTorch导入
import torch
import torch.nn.functional as F

max_threads = multitasking.config['CPU_CORES']
# max_threads=1
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='基于嵌入向量的itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test_emb.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'基于嵌入向量的itemcf 召回，mode: {mode}')


def cal_sim(df, emb_df):
    """
    基于文章嵌入向量计算物品间的相似度
    
    参数:
    df: 包含用户点击数据的DataFrame，至少包含'user_id'和'click_article_id'两列
    emb_df: 包含文章嵌入向量的DataFrame
    
    返回:
    sim_dict: 物品相似度字典，key为物品ID，value为与其它物品的相似度
    user_item_dict: 用户-物品倒排表，key为用户ID，value为该用户点击过的物品列表
    """
    # 按用户分组，收集每个用户点击过的物品列表
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))
    
    # 构建文章嵌入向量字典
    article_emb_dict = {}
    emb_columns = [col for col in emb_df.columns if col.startswith('emb_')]
    
    for _, row in tqdm(emb_df.iterrows(), total=len(emb_df), desc="加载文章嵌入向量"):
        article_id = int(row['article_id'])
        emb_vec = row[emb_columns].values.astype(float)
        article_emb_dict[article_id] = emb_vec
    
    log.debug(f"加载了 {len(article_emb_dict)} 个文章的嵌入向量")
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    log.debug(f"使用设备: {device}")
    if torch.cuda.is_available():
        log.debug(f"GPU数量: {gpu_count}")
        if gpu_count > 1:
            log.debug("将使用多GPU并行计算")
    
    # 存储物品间相似度的字典
    sim_dict = {}

    # 为每个文章计算与其他文章的相似度
    article_ids = list(article_emb_dict.keys())
    log.debug(f"开始计算文章相似度，共有 {len(article_ids)} 个文章")
    
    # 将所有嵌入向量转换为PyTorch张量
    all_embeddings = torch.tensor(
        np.array([article_emb_dict[aid] for aid in article_ids]), 
        dtype=torch.float32
    )
    
    # 定义批次大小以避免内存溢出
    batch_size = min(1024, len(article_ids))  # 固定批次大小，避免GPU内存问题
    log.debug(f"使用批次大小: {batch_size}")
    
    # 为每个文章找出最相似的200个文章
    log.debug("处理相似度结果...")
    for i in tqdm(range(0, len(article_ids), batch_size), desc="计算相似度批次"):
        # 获取当前批次的文章ID
        batch_end = min(i + batch_size, len(article_ids))
        batch_article_ids = article_ids[i:batch_end]
        
        # 将当前批次的嵌入向量移到GPU
        batch_embeddings = all_embeddings[i:batch_end].to(device)
        # 将所有嵌入向量移到GPU
        gpu_embeddings = all_embeddings.to(device)
        
        # 计算当前批次与所有文章的相似度
        # 先进行归一化
        batch_embeddings_norm = F.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings_norm = F.normalize(gpu_embeddings, p=2, dim=1)
        
        # 计算余弦相似度矩阵
        similarity_matrix = torch.mm(batch_embeddings_norm, all_embeddings_norm.t())
        
        # 将结果移回CPU并转换为numpy数组
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        # 处理当前批次的结果
        for j, article_id in enumerate(batch_article_ids):
            sim_dict[article_id] = {}
            
            # 获取与当前文章的相似度（排除自己）
            similarities = similarity_matrix[j]
            # 将自己与自己的相似度设为-1以排除
            self_index = article_ids.index(article_id)
            similarities[self_index] = -1
            
            # 获取前200个最相似的文章
            top_indices = np.argpartition(similarities, -200)[-200:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            for idx in top_indices:
                if idx != self_index:  # 确保不包含自己
                    related_article_id = article_ids[idx]
                    sim_dict[article_id][related_article_id] = float(similarities[idx])
        
        # 清理GPU内存
        del batch_embeddings, gpu_embeddings, batch_embeddings_norm, all_embeddings_norm, similarity_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 清理所有嵌入向量的GPU内存
    del all_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    # 遍历查询数据中的每一条记录
    for user_id, item_id in tqdm(df_query.values):
        # 如果用户没有历史行为则跳过
        if user_id not in user_item_dict:
            continue
        # 获取用户最近点击的2个物品
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        # 基于用户历史行为进行物品召回
        all_candidates = {}  # 用于存储所有候选物品及其分数
        for loc, item in enumerate(interacted_items):
            # 确保item在相似度字典中
            if item not in item_sim:
                continue
            # 获取与当前物品最相似的200个物品
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                # 排除用户已经点击过的物品
                if relate_item not in interacted_items:
                    # 累积相似度分数，不考虑位置权重
                    if relate_item not in all_candidates:
                        all_candidates[relate_item] = 0
                    all_candidates[relate_item] += wij

        # 对所有候选物品按相似度分数排序，取前100个
        sim_items = sorted(all_candidates.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # 构造结果DataFrame
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 设置标签：-1表示待预测项，其他情况设置正负样本标签
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            # 只有当正样本在召回列表中时才标记为1
            if item_id in item_ids:
                df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # 调整列顺序和数据类型
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)
    
    # 合并结果数据
    if data_list:  # 只有当有数据时才合并
        df_data = pd.concat(data_list, sort=False).reset_index(drop=True)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    # 保存中间结果
    os.makedirs('../user_data/tmp/itemcf_emb', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf_emb/{worker_id}.pkl')


if __name__ == '__main__':
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_path = os.path.join(project_root, 'user_data')
    data_path = os.path.join(user_data_path, 'data')
    
    # 根据模式选择不同的数据源
    if mode == 'valid':
        df_click = pd.read_pickle(os.path.join(data_path, 'offline/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'offline/query.pkl'))

        os.makedirs(os.path.join(user_data_path, 'sim/offline'), exist_ok=True)
        sim_pkl_file = os.path.join(user_data_path, 'sim/offline/itemcf_emb_sim.pkl')
    else:
        df_click = pd.read_pickle(os.path.join(data_path, 'online/click.pkl'))
        df_query = pd.read_pickle(os.path.join(data_path, 'online/query.pkl'))

        os.makedirs(os.path.join(user_data_path, 'sim/online'), exist_ok=True)
        sim_pkl_file = os.path.join(user_data_path, 'sim/online/itemcf_emb_sim.pkl')

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 加载文章嵌入向量
    articles_emb_path = os.path.join(project_root, 'data/articles_emb.csv')
    log.debug(f'加载文章嵌入向量: {articles_emb_path}')
    df_articles_emb = pd.read_csv(articles_emb_path)
    log.debug(f'文章嵌入向量维度: {df_articles_emb.shape}')

    # 计算物品相似度
    item_sim, user_item_dict = cal_sim(df_click, df_articles_emb)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回过程
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    tmp_path = os.path.join(user_data_path, 'tmp/itemcf_emb')
    os.makedirs(tmp_path, exist_ok=True)
    for path, _, file_list in os.walk(tmp_path):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    # 多线程执行召回任务
    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    # 等待所有任务完成
    multitasking.wait_for_tasks()
    log.info('合并任务')

    # 合并所有召回结果
    data_list = []
    for path, _, file_list in os.walk(tmp_path):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            data_list.append(df_temp)
    
    # 使用 pd.concat 替代 append，提高性能并避免弃用警告
    if data_list:
        df_data = pd.concat(data_list, sort=False).reset_index(drop=True)
    else:
        df_data = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')
    log.debug(f'df_data shape: {df_data.shape}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        
        # 打印一些统计数据以帮助调试
        log.debug(f"召回结果中的用户数: {df_data['user_id'].nunique()}")
        log.debug(f"召回结果总数: {len(df_data)}")
        log.debug(f"有标签的数据数: {len(df_data[df_data['label'].notnull()])}")
        
        # 确保query数据中包含正样本
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        log.debug(f"查询数据中的用户数: {total}")

        if len(df_data[df_data['label'].notnull()]) > 0:
            hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50, accuracy = evaluate(
                df_data[df_data['label'].notnull()], total)

            log.debug(
                f'itemcf_emb: \n'
                f'accuracy(准确率): \t{accuracy:.4f}\n'  # 成功处理的用户数占总用户数的比例
                f'hitrate_5(Top5命中率): \t{hitrate_5:.4f}\n'  # 推荐Top5中包含用户实际点击文章的比例
                f'mrr_5(Top5平均倒数排名): \t{mrr_5:.4f}\n'  # Top5的平均倒数排名
                f'hitrate_10(Top10命中率): \t{hitrate_10:.4f}\n'  # 推荐Top10中包含用户实际点击文章的比例
                f'mrr_10(Top10平均倒数排名): \t{mrr_10:.4f}\n'  # Top10的平均倒数排名
                f'hitrate_20(Top20命中率): \t{hitrate_20:.4f}\n'  # 推荐Top20中包含用户实际点击文章的比例
                f'mrr_20(Top20平均倒数排名): \t{mrr_20:.4f}\n'  # Top20的平均倒数排名
                f'hitrate_40(Top40命中率): \t{hitrate_40:.4f}\n'  # 推荐Top40中包含用户实际点击文章的比例
                f'mrr_40(Top40平均倒数排名): \t{mrr_40:.4f}\n'  # Top40的平均倒数排名
                f'hitrate_50(Top50命中率): \t{hitrate_50:.4f}\n'  # 推荐Top50中包含用户实际点击文章的比例
                f'mrr_50(Top50平均倒数排名): \t{mrr_50:.4f}'  # Top50的平均倒数排名
            )
        else:
            log.debug("没有找到带标签的数据，无法计算评估指标")

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle(os.path.join(data_path, 'offline/recall_itemcf_emb.pkl'))
    else:
        df_data.to_pickle(os.path.join(data_path, 'online/recall_itemcf_emb.pkl'))