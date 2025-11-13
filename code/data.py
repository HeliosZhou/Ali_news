import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

# 添加项目根目录路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Logger

# 随机数种子，让结果可以重现
random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
# 修改为基于项目根目录的路径
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    """
    离线数据处理函数
    
    该函数用于准备离线训练和验证所需的数据集。主要功能包括：
    1. 从训练数据中随机抽取一部分用户作为验证集用户
    2. 对验证集用户采用留一法策略，将其最后一次点击行为作为验证目标
    3. 构建包含训练集、验证集和测试集的完整数据集
    4. 将处理后的数据保存到指定目录
    
    参数:
    df_train_click: 训练集点击日志数据
    df_test_click: 测试集点击日志数据
    
    处理流程:
    - 随机采样50000个训练用户作为验证用户
    - 对验证用户采用留一法，最后一条记录作为验证标签，其余作为训练数据
    - 合并所有训练数据并按用户ID和时间戳排序
    - 为测试用户创建待预测的查询条目
    - 保存处理后的点击日志和查询数据
    """
    train_users = df_train_click['user_id'].values.tolist()
    log.debug(f'total train users: {len(train_users)}, unique users: {len(set(train_users))}')
    
    # 确保采样的用户数量不超过实际用户数量
    sample_size = min(50000, len(set(train_users)))
    val_users = sample(train_users, sample_size)
    val_users = list(set(val_users))  # 去重
    log.debug(f'val_users num: {len(set(val_users))}')
    log.debug(f'val_users sample (first 5): {val_users[:5]}')  # 查看采样的前5个用户
    log.debug(f'val_users dtype: {[type(x) for x in val_users[:5]]}')  # 查看采样用户的数据类型

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = []
    valid_query_list = []

    groups = df_train_click.groupby(['user_id'])
    matched_val_users = 0
    total_groups = 0
    
    for user_id, g in tqdm(groups):
        # 处理元组键的情况
        if isinstance(user_id, tuple):
            actual_user_id = user_id[0]
        else:
            actual_user_id = user_id
            
        total_groups += 1
        if actual_user_id in val_users:
            # 检查该用户是否有足够的行为数据
            if len(g) > 1:
                valid_query = g.tail(1)
                valid_query_list.append(
                    valid_query[['user_id', 'click_article_id']])

                train_click = g.head(g.shape[0] - 1)
                click_list.append(train_click)
                matched_val_users += 1
            else:
                # 如果用户只有一条行为数据，则全部放入训练集
                click_list.append(g)
        else:
            click_list.append(g)
            
    log.debug(f'total groups processed: {total_groups}')
    log.debug(f'matched val users: {matched_val_users}')
    df_train_click = pd.concat(click_list, sort=False)
    
    # 修复：检查valid_query_list是否为空
    if valid_query_list:
        df_valid_query = pd.concat(valid_query_list, sort=False)
    else:
        # 如果列表为空，创建一个空的DataFrame
        df_valid_query = pd.DataFrame(columns=['user_id', 'click_article_id'])
        log.debug(f'valid_query_list is empty, created an empty DataFrame.')

    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    # 修改为基于项目根目录的路径
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_path = os.path.join(project_root, 'user_data')
    data_path = os.path.join(user_data_path, 'data')
    
    os.makedirs(os.path.join(data_path, 'offline'), exist_ok=True)

    df_click.to_pickle(os.path.join(data_path, 'offline/click.pkl'))
    df_query.to_pickle(os.path.join(data_path, 'offline/query.pkl'))


def data_online(df_train_click, df_test_click):
    """
    在线数据处理函数
    
    该函数用于准备在线预测所需的数据集。主要功能包括：
    1. 合并训练集和测试集的所有点击日志数据
    2. 为测试集中的每个用户创建待预测的查询条目
    3. 将处理后的数据保存到指定目录
    
    参数:
    df_train_click: 训练集点击日志数据
    df_test_click: 测试集点击日志数据
    
    处理流程:
    - 提取测试集中的所有唯一用户
    - 为每个测试用户创建一个查询条目（文章ID设为-1表示待预测）
    - 合并训练集和测试集点击日志并按用户ID和时间戳排序
    - 保存处理后的点击日志和查询数据
    """
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    # 修改为基于项目根目录的路径
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_data_path = os.path.join(project_root, 'user_data')
    data_path = os.path.join(user_data_path, 'data')
    
    os.makedirs(os.path.join(data_path, 'online'), exist_ok=True)

    df_click.to_pickle(os.path.join(data_path, 'online/click.pkl'))
    df_query.to_pickle(os.path.join(data_path, 'online/query.pkl'))


if __name__ == '__main__':
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data')
    
    df_train_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
    df_test_click = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)