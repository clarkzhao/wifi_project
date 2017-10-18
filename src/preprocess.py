# -*- coding: utf-8 -*-

"""
The proprocess of original file is conducted. 
The output should be the desired data for training and validation, and test

Usage:
    please go the the directory ./src/
    cd src/
    python preprocess.py
    
    you can change constant KEEP_PER to modify the number of wifi strength feature to expand, 
    higher the value, lower the number of features to expand.
"""
from __future__ import division, print_function # Imports from __future__ since we're running Python 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEEP_PER = 1e-3

def filter_bssid(input_df, keep_percentage = KEEP_PER):
    '''
    输出要保留的wifi特征名，保存成list输出
    
    Args:
        input_df:
            输入数据集，pandas.DATAFRAME
        
        keep_percentage:
            bssid保留的最小比例，是某个bssid被扫描的总次数占全部bssid的被扫描的总次数
            该比例最小为1/全部bssid的总共被扫描次数，最多为全部数据集个数/全部bssid的总共被扫描次数
    Returns:
        bssids: 准备拓展特征空间的bssid list
    '''
    # 读取所以可被发现的wifi bssid，并将发现总次数存入wifi_counts中
    wifi_counts = {}
    for wifi_info in input_df['wifi_infos']:
        for bssid in wifi_info.split(';'):
            bssid = bssid.split('|')[0]
            if bssid not in wifi_counts:
                wifi_counts[bssid] = 1
            else:
                wifi_counts[bssid] += 1
    
    # 存放准备保留的wifi bssid
    bssids = []
    min_num_wifi = sum(wifi_counts.values())*keep_percentage
    for key,values in wifi_counts.items():
        if values > min_num_wifi:
            bssids.append(key)
    
    # i = 0
    # counts = False
    # for wifi_info in input_df['wifi_infos']:
    #     for bssid in wifi_info.split(';'):
    #         bssid, strength, _ = bssid.split('|')
    #         if bssid in bssids:
    #             counts = True
    #     if counts:
    #         i += 1
    # print("percentage of useful wifi scans: %", i/input_df.shape[0]*100)
    
    return bssids


def expand_wifi_feature(input_df, bssids):
    '''
    根据给定的bssid list和DataFrame，拓展特征空间
    input_df: Pandas.DataFrame

    bssids: 准备拓展的bssid list
    '''
    test = input_df.copy()
    # 建立全部强度值为-110的wifi特征
    wifi_features = np.zeros((test.shape[0], len(bssids)), dtype='int')
    wifi_features += -110
    wifi_features
    df_wifi = pd.DataFrame(wifi_features, columns = bssids)
    df_wifi.set_index(test.index, inplace=True)
    test = test.join(df_wifi)

    # 替换部分可被扫描的wifi信号强度为
    for i in test.index:
        wifi_info = test.loc[i, 'wifi_infos']   
        bssid2strength = {}
        for data in wifi_info.split(';'):
            bssid, strength, _ = data.split('|')
            bssid2strength[bssid] = int(strength)
        all_bssid = bssid2strength.keys()
        for bssid in all_bssid:
            if bssid in bssids:
                test.loc[i, bssid] = bssid2strength[bssid]
    return test


def standardize(input_df):
    '''
    标准化经纬度，wifi信号强度
    '''
    # 分别标准化经纬度
    cols_to_norm = ['longitude', 'latitude']
    input_df.loc[:, cols_to_norm] = input_df.loc[:, cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())
    
    # 整体标准化bssids
    cols_to_norm = bssids
    mean = np.mean(input_df.loc[:, cols_to_norm].as_matrix())
    std = np.std(input_df.loc[:, cols_to_norm].as_matrix())
    input_df.loc[:, cols_to_norm] = (input_df.loc[:, cols_to_norm] - mean)/std
    input_df.loc[:, cols_to_norm] = input_df[bssids].astype('float16') #降低空间需求

def toOneHot(input_df):
    """将string格式的class，shop_id，编码成1-of-k的形式
    Args:
        input_df: pandas.DATAFRAME
    
    Returns:
        one hot encoded targets
    """
    targets = input_df.loc[:,'shop_id'].values
    num_classes = 0 
    shop_list = []
    for shop_id in targets:
        if shop_id not in shop_list:
            shop_list.append(shop_id)
            num_classes += 1
    one_of_k_targets = np.zeros((targets.shape[0], num_classes))
    for i in range(targets.shape[0]):
        one_of_k_targets[i, shop_list.index(targets[i])] = 1
    return one_of_k_targets

if __name__ == '__main__':

    print("=========== Starts loading original data ===========")
    shop_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ccf_first_round_shop_info.csv')
    user_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ccf_first_round_user_shop_behavior.csv')
    eval_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'evaluation_public.csv')

    shop_train = pd.read_csv(shop_path, delimiter=',')
    user_train = pd.read_csv(user_path, delimiter=',')
    evaluation_data = pd.read_csv(eval_path, delimiter=',')

    print("=========== Successfully loading original data ===========")
    print("The shape of the original data of user shop behavior is: {0} \n".format(user_train.shape),
        "The shape of the original data for evaluation is: {0}".format(evaluation_data.shape))

    print("=========== Stats detecting the total number of malls ===========")
    # 检测有多少个商场，以及他们的名字
    unique_mall = []
    idx = 0
    for i in range(len(shop_train['mall_id'])):
        if shop_train['mall_id'][i] not in unique_mall:
            unique_mall.append(shop_train['mall_id'][i])
            idx += 1
    print("There are a total number of {0} malls and they are:".format(len(unique_mall)))
    print(unique_mall)


    print("=========== Starts creating mall feature in the original data ===========")
    # 建立shop_id 到mall_id的映射
    shop2mall = shop_train.set_index('shop_id').to_dict()['mall_id']
    # 在user_train数据集中，根据shop2mall新建一个叫mall_id的特征
    mall_id = []
    for shop_id in user_train['shop_id']:
        mall_id.append(shop2mall[shop_id])
    # 在user_train中增加一个feature，mall_id，新的训练集叫user_train_mall_added
    user_train_mall_added = user_train.assign(mall_id=mall_id)
    print("The shape of the data after assigning the mall id  is: {0} \n".format(user_train_mall_added.shape))


    print("=========== Starts Slicing the original data by mall_id ===========")
    # 根据mall_id切割训练集
    user_train_grouped = user_train_mall_added.groupby('mall_id')
    train_sets = []
    print("mall id, shape")
    for mall_id in unique_mall:
        train_sets.append(user_train_grouped.get_group(mall_id))
        print(mall_id, train_sets[-1].shape)
    
    print("=========== Starts preprocess and output data as CSV file for training and validation  ===========")
    # 对每个训练集输出成csv
    for df in train_sets[1:2]:
        bssids = filter_bssid(df)
        df = expand_wifi_feature(df, bssids)
        standardize(df)
        path = os.path.join(os.path.dirname(os.getcwd()), 'data', df.iloc[-1,6]+'.csv')
        df.to_csv(path)
        print("Successfully stored data as {0} with shape of {1}".format(path, df.shape))