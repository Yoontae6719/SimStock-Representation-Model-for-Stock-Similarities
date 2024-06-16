import pandas as pd
import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

import tslearn
import tslearn.metrics
from sklearn.preprocessing import MinMaxScaler
import warnings
# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from fastdtw import fastdtw
from tqdm import tqdm

def get_index_and_col(result, data_df):
    result = pd.DataFrame(result)
    result.columns = data_df.columns
    result.index = data_df.columns
    return result

def calculate_fastdtw_distance(i, j, data_transposed):
    return fastdtw(data_transposed[i].copy(), data_transposed[j].copy())[0]

def calculate_distance_matrix(data, data_df, scale = False, need_dtw = True):
    
    
    if scale == True:
        scaler = MinMaxScaler() 
        data = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns, index=data_df.index).to_numpy()
        data_transposed = data.T
    else:
        data_transposed = data.T
        
    num_stocks = data_transposed.shape[0]
    indices = [(i, j) for i in range(num_stocks) for j in range(i, num_stocks)]
    
    # Calculate fastDTW distance matrix using parallelization
    if need_dtw == True:
        fastdtw_distances = Parallel(n_jobs=-1)(delayed(calculate_fastdtw_distance)(i, j, data_transposed) for i, j in tqdm(indices, desc="Calculating fastDTW distances"))
        distance_matrix_fastDTW = np.zeros((num_stocks, num_stocks))
        for (i, j), dist in zip(tqdm(indices, desc="Filling fastDTW distance matrix"), fastdtw_distances):
            distance_matrix_fastDTW[i, j] = dist
            distance_matrix_fastDTW[j, i] = dist
    else:
        distance_matrix_fastDTW = np.zeros((num_stocks, num_stocks))


    corr_maxtix = data_df.corr()

    return (
        get_index_and_col(distance_matrix_fastDTW, data_df),
        corr_maxtix
    )

def get_simstock_result(data_name):
    us = pd.read_csv("../main_result_ex_fund/{}.csv".format(data_name)) #nasdaq_nasdaq_2022
    us = us[["Date","Stock_","Label"]]
    df_grouped = us.groupby('Stock_')['Label'].apply(list).reset_index()
    df_expanded = pd.DataFrame(df_grouped['Label'].tolist(), index=df_grouped['Stock_']).transpose()
    df_expanded = df_expanded.iloc[:us[us["Stock_"] == us["Stock_"].unique()[0]].shape[0], :]
    df_expanded.index = us[us["Stock_"] == us["Stock_"].unique()[0]].set_index("Date").index
    df_expanded = df_expanded.dropna(axis = 1)
    return df_expanded

def get_ts2vec_result(data_name,label):
    us = pd.read_csv("./main_result_ts2vec/{}.csv".format(data_name)) #nasdaq_nasdaq_2022
    us = us[["Date","Stock_",label]]
    df_grouped = us.groupby('Stock_')[label].apply(list).reset_index()
    df_expanded = pd.DataFrame(df_grouped[label].tolist(), index=df_grouped['Stock_']).transpose()
    df_expanded = df_expanded.iloc[:us[us["Stock_"] == us["Stock_"].unique()[0]].shape[0], :]
    df_expanded.index = us[us["Stock_"] == us["Stock_"].unique()[0]].set_index("Date").index
    df_expanded = df_expanded.dropna(axis = 1)
    return df_expanded

def get_corr1_corr2(data_list):    
    corr1 = yf.download(data_list, start='2018-01-01', end='2022-12-31')['Close']
    corr2 = yf.download(data_list, start='2022-01-01', end='2022-12-31')['Close']
    corr1 = corr1.corr()
    corr2 = corr2.corr()
    return corr1, corr2


def calculate_diagonal_perform(corr1, corr2, simstock, peer, ts2vec, gt_corr, gt_dtw, num_stocks):
    all_stock_list = list(corr1.columns)
    result_corr_list = []
    result_dtw_list = []
    peer_dict = {row["symbol"]: [item.strip() for item in row["peers"].split(",")][:num_stocks] for _, row in peer.iterrows()}

    corr1.index = corr1.columns
    corr2.index = corr2.columns
    result_corr_list = []
    result_dtw_list = []
    for i in tqdm(all_stock_list, desc="Processing stocks"):
        corr1_list_ = list(corr1[i].nlargest(num_stocks).index)
        corr2_list_ = list(corr2[i].nlargest(num_stocks).index)
        simstock_corr_list_ = list(simstock[i].nlargest(num_stocks).index)
        ts2vec_corr_list_ = list(ts2vec[i].nlargest(num_stocks).index)

        peer_list = peer_dict.get(i, [])
        peer_list = peer_list[:num_stocks-1]

        peer_gt_corr_index = list(set(peer_list) & set(gt_corr.index))
        peer_gt_dtw_index = list(set(peer_list) & set(gt_dtw.index))

        result_corr = {
                'symbol': i,
                'corr1_mean': round(gt_corr.loc[corr1_list_][i].values[1:].mean(), 3),
                'corr2_mean': round(gt_corr.loc[corr2_list_][i].values[1:].mean(), 3),
                'peer_mean': round(np.nanmean(gt_corr.loc[peer_gt_corr_index][i].values), 3) if peer_gt_corr_index else np.nan,
                'simstock_mean': round(gt_corr.loc[gt_corr.columns.intersection(simstock_corr_list_)][i].values[1:].mean(), 3),
                'ts2vec_mean': round(gt_corr.loc[gt_corr.columns.intersection(ts2vec_corr_list_)][i].values[1:].mean(), 3),

                'corr1_std': round(gt_corr.loc[corr1_list_][i].values[1:].std(), 3),
                'corr2_std': round(gt_corr.loc[corr2_list_][i].values[1:].std(), 3),
                'peer_std': round(np.nanstd(gt_corr.loc[peer_gt_corr_index][i].values), 3) if peer_gt_corr_index and len(peer_gt_corr_index) > 1 else np.nan,
                'simstock_std': round(gt_corr.loc[gt_corr.columns.intersection(simstock_corr_list_)][i].values[1:].std(), 3),
                'ts2vec_std': round(gt_corr.loc[gt_corr.columns.intersection(ts2vec_corr_list_)][i].values[1:].std(), 3),

                }
        result_corr_list.append(result_corr)

        result_dtw = {
                'symbol': i,
                'corr1_mean': round(gt_dtw.loc[corr1_list_][i].values[1:].mean(), 3),
                'corr2_mean': round(gt_dtw.loc[corr2_list_][i].values[1:].mean(), 3),
                'peer_mean': round(np.nanmean(gt_dtw.loc[peer_gt_corr_index][i].values), 3) if peer_gt_corr_index else np.nan,
                'simstock_mean': round(gt_dtw.loc[gt_dtw.columns.intersection(simstock_corr_list_)][i].values[1:].mean(), 3),
                'ts2vec_mean': round(gt_dtw.loc[gt_dtw.columns.intersection(ts2vec_corr_list_)][i].values[1:].mean(), 3),

                'corr1_std': round(gt_dtw.loc[corr1_list_][i].values[1:].std(), 3),
                'corr2_std': round(gt_dtw.loc[corr2_list_][i].values[1:].std(), 3),
                'peer_std': round(np.nanstd(gt_dtw.loc[peer_gt_corr_index][i].values), 3) if peer_gt_corr_index and len(peer_gt_corr_index) > 1 else np.nan,
                'simstock_std': round(gt_dtw.loc[gt_dtw.columns.intersection(simstock_corr_list_)][i].values[1:].std(), 3),
                'ts2vec_std': round(gt_dtw.loc[gt_dtw.columns.intersection(ts2vec_corr_list_)][i].values[1:].std(), 3),

                }
        result_dtw_list.append(result_dtw)
        result_corr = pd.DataFrame(result_corr_list)
        result_dtw = pd.DataFrame(result_dtw_list)
        
    result_corr = pd.DataFrame(result_corr_list)
    result_dtw = pd.DataFrame(result_dtw_list)

    result_corr["top_k"] = num_stocks-1
    result_dtw["top_k"] = num_stocks-1
    
    return result_corr, result_dtw


def get_diagonal_perform(corr1,
                         corr2,
                         simstock,
                         peer,
                         ts2vec,
                         gt_corr, gt_dtw):
    corr_1_us_us, dtw_1_us_us =  calculate_diagonal_perform(corr1 = corr1,
                                                                               corr2 = corr2,
                                                                            simstock = simstock,
                                                                                peer = peer,
                                                                               ts2vec = ts2vec,
                                                                             gt_corr = gt_corr,
                                                                              gt_dtw = gt_dtw, 
                                                                          num_stocks = 2 # top 1
                                                                             )


    corr_3_us_us, dtw_3_us_us =  calculate_diagonal_perform(corr1 = corr1,
                                                                               corr2 = corr2,
                                                                            simstock = simstock,
                                                                                peer = peer,
                                                                               ts2vec = ts2vec,
                                                                             gt_corr = gt_corr,
                                                                              gt_dtw = gt_dtw, 
                                                                          num_stocks = 4 # top 3
                                                                             )

    corr_5_us_us, dtw_5_us_us=  calculate_diagonal_perform(corr1 = corr1,
                                                                               corr2 = corr2,
                                                                            simstock = simstock,
                                                                                peer = peer,
                                                                               ts2vec = ts2vec,
                                                                             gt_corr = gt_corr,
                                                                              gt_dtw = gt_dtw, 

                                                                          num_stocks = 6 # top 5
                                                                             )

    corr_7_us_us, dtw_7_us_us =  calculate_diagonal_perform(corr1 = corr1,
                                                                               corr2 = corr2,
                                                                            simstock = simstock,
                                                                                peer = peer,
                                                                               ts2vec = ts2vec,
                                                                             gt_corr = gt_corr,
                                                                              gt_dtw = gt_dtw, 

                                                                          num_stocks = 8 # top 7
                                                                             )

    corr_9_us_us, dtw_9_us_us =  calculate_diagonal_perform(corr1 = corr1,
                                                                               corr2 = corr2,
                                                                            simstock = simstock,
                                                                                peer = peer,
                                                                               ts2vec = ts2vec,
                                                                             gt_corr = gt_corr,
                                                                              gt_dtw = gt_dtw, 

                                                                          num_stocks = 10 # top 9
                                                                             )

    corr_us_us_full = pd.concat([corr_1_us_us, corr_3_us_us, corr_5_us_us, corr_7_us_us, corr_9_us_us]).reset_index(drop = True)
    dtw_us_us_full = pd.concat([dtw_1_us_us, dtw_3_us_us, dtw_5_us_us, dtw_7_us_us, dtw_9_us_us]).reset_index(drop = True)
    
    return corr_us_us_full, dtw_us_us_full


