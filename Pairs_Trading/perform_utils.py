import pandas as pd
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from fastdtw import fastdtw
from tqdm import tqdm
import tslearn
import tslearn.metrics
from sklearn.preprocessing import MinMaxScaler
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS

import yfinance as yf

import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def find_hyperparam(tic_list, z_score_value, baseline, train_period_start, train_period_end, stop_loss):
    
    tic_list_ = tic_list[1:]
    
    target_data = yf.download(tic_list[0], start=train_period_start, end=train_period_end,progress=False)["Adj Close"] 
    if len(tic_list) == 2: 
        pair_data = yf.download(tic_list_, start=train_period_start, end=train_period_end,progress=False)["Adj Close"]
        pair_data = pd.DataFrame({tic_list[1] : pair_data})
    
    else:
        pair_data = yf.download(tic_list_, start=train_period_start, end=train_period_end,progress=False)["Adj Close"] # Default top 10
        
    target_data = target_data.resample('D').last().fillna(method='ffill')
    pair_data = pair_data.resample('D').last().fillna(method='ffill')
    target_data, pair_data = target_data.align(pair_data, join='inner')


    pair_portfolio_perform_ = pd.DataFrame()
    
    get_param = pd.DataFrame({"window1": np.nan, "window2": np.nan, "baseline" : np.nan, "best_wealth" : np.nan}, index = tic_list_)
    

    for i in range(len(tic_list_)):
        best_window1 = None
        best_window2 = None
        best_wealth = -np.inf
        for window1_value in range(2, 15):
            for window2_value in range(window1_value * 1 + 1, window1_value * 4 + 1):
                try:
                    df = trading_simulation(method = "price_ratio",
                                           z_score_value = z_score_value ,
                                           Asset1= target_data,
                                           Asset2= pair_data[tic_list_[i]],
                                           window1= window1_value,
                                           window2 =window2_value, leverage= 10_000, stop_loss= stop_loss)
                    result = get_perform(df, baseline, tic_list, i)
                    wealth = result.loc["Wealth"].values[0]
                    
                    if wealth > best_wealth:
                        
                        best_wealth = wealth
                        best_window1 = window1_value
                        best_window2 = window2_value
                except:
                    continue
        get_param.iloc[i, :] = [best_window1, best_window2, baseline, best_wealth]
                
    return get_param

def test_pair_trading(param_df, tic_list, z_score_value, baseline, stop_loss):
    all_reulst = None
    all_reulst_mean = None
    
    tic_list_ = tic_list[1:]
    target_data = yf.download(tic_list[0],  "2023-01-01", end="2024-01-01",progress=False)["Adj Close"] 
    if len(tic_list) == 2: 
        pair_data = yf.download(tic_list_, start=  "2023-01-01", end="2024-01-01",progress=False)["Adj Close"]
        pair_data = pd.DataFrame({tic_list[1] : pair_data})
    
    else:
        pair_data = yf.download(tic_list_, start=  "2023-01-01", end="2024-01-01",progress=False)["Adj Close"] # Default top 10

    target_data = target_data.resample('D').last().fillna(method='ffill')
    pair_data = pair_data.resample('D').last().fillna(method='ffill')
    target_data, pair_data = target_data.align(pair_data, join='inner')

        
    pair_portfolio_perform_ = pd.DataFrame()
    
    for i in range(len(tic_list_)):
        try:
            df = trading_simulation(method = "price_ratio",
                                     z_score_value = z_score_value ,
                                     Asset1= target_data,
                                     Asset2=  pair_data[param_df.index[i]],
                                     window1= int(param_df.loc[param_df.index[i]][:2].values[0]),  
                                     window2 =int(param_df.loc[param_df.index[i]][:2].values[1]), leverage= 10_000, stop_loss= stop_loss)
            result = get_perform(df, baseline, tic_list, i)
            pair_portfolio_perform_ = pd.concat([pair_portfolio_perform_, result], axis = 1)
        except:
            pair_portfolio_perform_ = pd.DataFrame()
            continue
    all_reulst = pair_portfolio_perform_
    
    all_reulst_mean = pair_portfolio_perform_.mean(1)
    all_reulst_std = pair_portfolio_perform_.std(1)
    all_reulst_max = pair_portfolio_perform_.max(1)
    all_reulst_min = pair_portfolio_perform_.min(1)
    
    return all_reulst, (all_reulst_mean, all_reulst_std, all_reulst_max, all_reulst_min), None

def run_pair(tic_list, z_score_value, baseline, train_period_start, train_period_end, stop_loss):
    param_set= find_hyperparam(tic_list,z_score_value, baseline, train_period_start, train_period_end, stop_loss)
    main_result, main_stat_, last_num = test_pair_trading(param_set, tic_list,z_score_value, baseline, stop_loss)
    return main_result, main_stat_, last_num

def all_run_pair(SS, TS, C1, C2,z_score_value, top_k, train_period_start, train_period_end, stop_loss):
    SS_main_result, SS_main_stat_, SS_last_num = run_pair(SS[:top_k], z_score_value, "SS", train_period_start, train_period_end, stop_loss)
    TS_main_result, TS_main_stat_, TS_last_num = run_pair(TS[:top_k], z_score_value, "TS", train_period_start, train_period_end, stop_loss)
    C1_main_result, C1_main_stat_, C1_last_num = run_pair(C1[:top_k], z_score_value,  "CORR1", train_period_start, train_period_end, stop_loss)
    C2_main_result, C2_main_stat_, C2_last_num = run_pair(C2[:top_k], z_score_value,  "CORR2", train_period_start, train_period_end, stop_loss)
    
    
    mean_ = pd.DataFrame({"SS_{}_mean".format(top_k-1) : SS_main_stat_[0], 
                          "TS_{}_mean".format(top_k-1) : TS_main_stat_[0], 
                          "C1_{}_mean".format(top_k-1) : C1_main_stat_[0], 
                          "C2_{}_mean".format(top_k-1) : C2_main_stat_[0], 
                     })
    
    std_ = pd.DataFrame({ "SS_{}_std".format(top_k-1) : SS_main_stat_[1], 
                          "TS_{}_std".format(top_k-1) : TS_main_stat_[1], 
                          "C1_{}_std".format(top_k-1) : C1_main_stat_[1], 
                          "C2_{}_std".format(top_k-1) : C2_main_stat_[1], 
                     })
    
    max_ = pd.DataFrame({ "SS_{}_max".format(top_k-1) : SS_main_stat_[2], 
                          "TS_{}_max".format(top_k-1) : TS_main_stat_[2], 
                          "C1_{}_max".format(top_k-1) : C1_main_stat_[2], 
                          "C2_{}_max".format(top_k-1) : C2_main_stat_[2], 
                     })
    
    min_ = pd.DataFrame({ "SS_{}_min".format(top_k-1) : SS_main_stat_[3], 
                          "TS_{}_min".format(top_k-1) : TS_main_stat_[3], 
                          "C1_{}_min".format(top_k-1) : C1_main_stat_[3], 
                          "C2_{}_min".format(top_k-1) : C2_main_stat_[3], 
                     })
    
    return (SS_main_result, TS_main_result, C1_main_result, C2_main_result), (mean_, std_,  max_, min_), (SS_last_num, TS_last_num,  C1_last_num, C2_last_num),



def all_run_pair_top_(SS, TS, C1, C2, train_period_start, train_period_end, stop_loss):
    #CVS_us_9_result, CVS_us_9_stat_, CVS_us_9_num_     =  all_run_pair(SS,
    #                                                                   TS,
    #                                                                   C1,
    #                                                                   C2,
    #                                                                 1.25, 10,train_period_start, train_period_end, stop_loss)
#
    #CVS_us_7_result, CVS_us_7_stat_, CVS_us_7_num_     =  all_run_pair(SS,
    #                                                                   TS,
    #                                                                   C1,
    #                                                                   C2,
    #                                                                 1.25, 8,train_period_start, train_period_end, stop_loss)
#

    CVS_us_5_result, CVS_us_5_stat_, CVS_us_5_num_     = all_run_pair(SS,
                                                                       TS,
                                                                       C1,
                                                                       C2,
                                                                     1.25, 6, train_period_start, train_period_end, stop_loss)

    CVS_us_3_result, CVS_us_3_stat_, CVS_us_3_num_     =  all_run_pair(SS,
                                                                       TS,
                                                                       C1,
                                                                       C2,
                                                                     1.25, 4, train_period_start, train_period_end, stop_loss)

    CVS_us_1_result, CVS_us_1_stat_, CVS_us_1_num_     =  all_run_pair(SS,
                                                                       TS,
                                                                       C1,
                                                                       C2,
                                                                     1.25, 2,train_period_start, train_period_end, stop_loss)
    return (CVS_us_5_result, CVS_us_3_result,CVS_us_1_result),\
           (CVS_us_5_stat_,CVS_us_3_stat_,CVS_us_1_stat_),\
           (CVS_us_5_num_,CVS_us_3_num_,CVS_us_1_num_)
    
    #return (CVS_us_9_result, CVS_us_7_result,CVS_us_5_result, CVS_us_3_result,CVS_us_1_result),\
    #       (CVS_us_9_stat_,CVS_us_7_stat_,CVS_us_5_stat_,CVS_us_3_stat_,CVS_us_1_stat_),\
    #       (CVS_us_9_num_,CVS_us_7_num_,CVS_us_5_num_,CVS_us_3_num_,CVS_us_1_num_)



def all_run_pair_top_exchange(SS_us, SS_sse, SS_szse, SS_tse,
                              TS_us, TS_sse, TS_szse, TS_tse,
                              C1_us, C1_sse, C1_szse, C1_tse,
                              C2_us, C2_sse, C2_szse, C2_tse, train_period_start, train_period_end, stop_loss):
    
    us_r,  us_s, us_n = all_run_pair_top_(SS_us,
                                            TS_us,
                                            C1_us,
                                            C2_us,  train_period_start, train_period_end, stop_loss)

    sse_r, sse_s, sse_n = all_run_pair_top_(SS_sse,
                                               TS_sse,
                                               C1_sse,
                                               C2_sse,  train_period_start, train_period_end, stop_loss)

    szse_r ,szse_s, szse_n = all_run_pair_top_(SS_szse,
                                                  TS_szse,
                                                  C1_szse,
                                                  C2_szse,  train_period_start, train_period_end, stop_loss)

    tse_r, tse_s, tse_n = all_run_pair_top_(SS_tse,
                                                        TS_tse,
                                                        C1_tse,
                                                        C2_tse,  train_period_start, train_period_end, stop_loss)
    
    return (us_r, sse_r, szse_r, tse_r), (us_s, sse_s, szse_s, tse_s)


def cointegration_checker(stock_dataframe, query, p):
    cointegrated_pairs = []
    
    k = stock_dataframe.shape[1]
    p_values = np.ones( (k, 1) )
    keys = stock_dataframe.keys()
    
    for j in range(0, 11):
            
        Asset_1 = stock_dataframe[query]
        Asset_2 = stock_dataframe[keys[j]]
            
        #iterating through the df and testing cointegration for all pairs of tickers
        Coint_Test = coint(Asset_1, Asset_2)
            
        pvalue = Coint_Test[1]
        # statsmodels coint returns p-values (our primary concern) in the 1th index slot
        p_values[j] = pvalue
        #p value matrix where the output of the coint test is the ith, jth index
        if pvalue < p:
            cointegrated_pairs.append( keys[j])
    
    cointegrated_pairs.remove(query)
    cointegrated_pairs = [query] + cointegrated_pairs
    return cointegrated_pairs


def get_perform(df, method, query_list, i):
    df_port_performances = pd.DataFrame(columns=[

        'Maximum Drawdown (%)',
        'Monthly 95% VaR (%)',
#        'Shape Ratio',
        'Wealth'
    ])

    wealth = df.iloc[-1].values[0] - 10_000
    df['daily_return'] = df['Cumulative Value'].pct_change()
    df['daily_return'] = df['daily_return'].dropna()

    # Annualized Return
    annualized_return = (1 + df['daily_return'].mean())**252 - 1

    # Annualized Std
    annualized_std = df['daily_return'].std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_std

    # Arithmetic Return
    arithmetic_mean_return = df['daily_return'].mean()
    arithmetic_return = arithmetic_mean_return * 252

    # Geometric Return
    geometric_mean_return = (df['Cumulative Value'].iloc[-1] / df['Cumulative Value'].iloc[0]) ** (1 / (len(df) - 1)) - 1
    geometric_return = geometric_mean_return * 252

    # Maximum Drawdown
    max_drawdown = -(df['Cumulative Value'] / df['Cumulative Value'].cummax() - 1).min()

    # Annualized Skewness
    daily_skewness = skew(df['daily_return'].dropna())
    annualized_skewness = daily_skewness * np.sqrt(252)

    # Annualized Kurtosis
    daily_kurtosis = kurtosis(df['daily_return'].dropna(), fisher=False)
    annualized_kurtosis = daily_kurtosis * np.sqrt(252)

    # Cumulative Return
    cumulative_return = (df['Cumulative Value'].iloc[-1] / df['Cumulative Value'].iloc[0]) - 1

    # Monthly 95% VaR
    monthly_var = -df['daily_return'].quantile(0.05)

    df_port_performances = df_port_performances.append({
#        'Arithmetic Return (%)': (arithmetic_return * 100).round(3),
#        'Geometric Return (%)': (geometric_return * 100).round(3),
#        'Cumulative Return (%)': (cumulative_return * 100).round(3),
#        'Annualized STD (%)': (annualized_std * 100).round(3),
#        'Annualized Skewness': annualized_skewness.round(3),
#        'Annualized Kurtosis': annualized_kurtosis.round(3),
        'Maximum Drawdown (%)': (max_drawdown * 100).round(3),
        'Monthly 95% VaR (%)': (monthly_var * 100).round(3),
#        'Shape Ratio': sharpe_ratio.round(3),
        'Wealth': wealth.round(3)
    }, ignore_index=True)

    df_port_performances = df_port_performances.T
    query_name = query_list[i+1]
    df_port_performances.columns = [f"{method}_{query_list[0]}_{query_name}_{i+1}"]

    return df_port_performances


def trading_simulation(method, z_score_value,Asset1, Asset2, window1, window2, leverage, stop_loss):
    transaction_cost = 0.001 # 10 BASIS POINTS
    if method == "price_ratio":
        price_ratio = Asset1/Asset2
    elif method == "spread":
        Asset_11 = Asset1.copy()
        Asset_22 = Asset2.copy()    
        Asset_11  = add_constant(Asset_11)
        results = OLS(Asset_22, Asset_11).fit()
        tic = pd.DataFrame(Asset_11).columns[0]
        coef = results.params[tic]
        Asset_11= Asset_11[tic]
        price_ratio = Asset_22 - (coef * Asset_11)
        

    moving_average1 = price_ratio.rolling(window=window1, center=False).mean()
    moving_average2 = price_ratio.rolling(window=window2, center=False).mean()
    std = price_ratio.rolling(window=window2).std()
    
    z_score = ((moving_average1-moving_average2)/std)

    
    # calculating the z score with moving averages as shown in previous sections
    profit, profit_high, profit_low = 0,0,0
    ratio_high_sell, ratio_high_buy, ratio_low_buy, ratio_low_sell = 0, 0, 0, 0
    low_trade_total, high_trade_total, = 0,0
    count_high, count_low = 0,0
    old_profit_high, old_profit_low = 0,0
    high_hit_rate, low_hit_rate = [], []
    low_dic, high_dic = {}, {}
    Asset1_shares, Asset2_shares = 0,0
    open_trade = 0
    potential_loss_high, potential_loss_low  = 0,0
    
    df = pd.DataFrame(columns = ['Date', 'Profit'])
    
    for i in range(len(price_ratio)):
        
        
        if z_score[i] > z_score_value and open_trade == 0: 
        #if the price ratio z score is high we will 'sell' the ratio 
        
            #calculating the maximum number of shares for each stock; positions are weighted equally
            Asset1_shares = (leverage // 2) // Asset1[i]
            Asset2_shares = (leverage // 2) // Asset2[i]
            
            
            #selling the ratio means you sell the higher stock (STT) and buy the lower stock (C)
            ratio_high_sell = Asset1[i] * Asset1_shares
            ratio_high_buy = Asset2[i] * Asset2_shares
            
                
            #tracking the number of trades for profit calculation and overall frequency
            count_high += 1
            high_trade_total +=1
            #checker count high with tracker
            open_trade = 1

##############################################################################################################################            
        elif z_score[i] < -z_score_value and open_trade == 0:
        #if the price ratio z score is low we will 'buy' the ratio 
        
            #calculating the maximum number of shares for each stock; positions are weighted equally
            Asset1_shares = (leverage // 2) // Asset1[i]
            Asset2_shares = (leverage // 2) // Asset2[i]
            
            #'Buying' the ratio means you buy the higher stock (STT) and sell the lower stock (C)
            ratio_low_buy = Asset1[i] * Asset1_shares
            ratio_low_sell = Asset2[i] * Asset2_shares
            
            count_low += 1
            low_trade_total +=1
            open_trade = 1

##############################################################################################################################            
        if open_trade == 1 and (abs(z_score[i]) > .5):

            potential_loss_high = ((ratio_high_sell - (Asset1[i] * Asset1_shares * count_high)) + ((Asset2[i]*Asset2_shares*count_high) - ratio_high_buy))
            potential_loss_low = (((Asset1[i]*Asset1_shares*count_low) - ratio_low_buy) + (ratio_low_sell - (Asset2[i]*Asset2_shares*count_low)))
            #tracking the current profit from high / low positions 
            
            if potential_loss_high < -stop_loss or potential_loss_low < -stop_loss:
            # if potential losses exceed stop loss then we will cut the positions
            
                old_profit_high = profit_high
                old_profit_low = profit_low
            
            
                #profit_high += ratio_high_sell - (Asset1[i] * Asset1_shares * count_high)
                #profit_high += (Asset2[i]*Asset2_shares*count_high) - ratio_high_buy
                
                profit_high += (ratio_high_sell - (Asset1[i] * Asset1_shares * count_high)) * (1 - transaction_cost)
                profit_high += ((Asset2[i] * Asset2_shares * count_high) - ratio_high_buy) * (1 - transaction_cost)

                
                if (profit_high-old_profit_high) != 0:
                    high_hit_rate.append(profit_high-old_profit_high)
                    high_dic[Asset1.index[i].strftime('%Y-%m-%d')] = (profit_high-old_profit_high)
                
                #profit_low += (Asset1[i]*Asset1_shares*count_low) - ratio_low_buy
                #profit_low += ratio_low_sell - (Asset2[i]*Asset2_shares*count_low)
                profit_low += ((Asset1[i] * Asset1_shares * count_low) - ratio_low_buy) * (1 - transaction_cost)
                profit_low += (ratio_low_sell - (Asset2[i] * Asset2_shares * count_low)) * (1 - transaction_cost)
                
                if (profit_low-old_profit_low) != 0:
                    low_hit_rate.append(profit_low-old_profit_low)
                    low_dic[Asset1.index[i].strftime('%Y-%m-%d')] = (profit_low-old_profit_low)
            
                ratio_high_sell, ratio_high_buy, ratio_low_buy, ratio_low_sell = 0, 0, 0, 0
                count_high,count_low = 0,0
                open_trade = 0
                        
                
            
        elif (abs(z_score[i]) < .5):
        #once the z score has returned to 'normal' we will close our positions
            
            #tracking the previous profit level so that we can calculate changes
            old_profit_high = profit_high
            old_profit_low = profit_low
            
            
            #profit_high += ratio_high_sell - (Asset1[i] * Asset1_shares * count_high)
            #profit_high += (Asset2[i]*Asset2_shares*count_high) - ratio_high_buy
            profit_high += (ratio_high_sell - (Asset1[i] * Asset1_shares * count_high)) * (1 - transaction_cost)
            profit_high += ((Asset2[i] * Asset2_shares * count_high) - ratio_high_buy) * (1 - transaction_cost)
            
            
            # profit is derived from (shorted share price - current price) + (current share price - initial long share price)
            
            if (profit_high-old_profit_high) != 0:
            #tracking profit from high trades for metrics
                high_hit_rate.append(profit_high-old_profit_high)
                high_dic[Asset1.index[i].strftime('%Y-%m-%d')] = (profit_high-old_profit_high)
                
            #profit_low += (Asset1[i]*Asset1_shares*count_low) - ratio_low_buy
            #profit_low += ratio_low_sell - (Asset2[i]*Asset2_shares*count_low)
            profit_low += ((Asset1[i] * Asset1_shares * count_low) - ratio_low_buy) * (1 - transaction_cost)
            profit_low += (ratio_low_sell - (Asset2[i] * Asset2_shares * count_low)) * (1 - transaction_cost)

            
            if (profit_low-old_profit_low) != 0:
            #tracking profit from low trades for metrics
                low_hit_rate.append(profit_low-old_profit_low)
                low_dic[Asset1.index[i].strftime('%Y-%m-%d')] = (profit_low-old_profit_low)
            
            #clearing all positions
            ratio_high_sell, ratio_high_buy, ratio_low_buy, ratio_low_sell = 0, 0, 0, 0
            count_high,count_low = 0,0
            open_trade = 0
            
            
    profit = profit_low + profit_high
    
    high_biggest_loss = min(high_hit_rate)
    high_biggest_gain = max(high_hit_rate)
    
    low_biggest_loss = min(low_hit_rate)
    low_biggest_gain = max(low_hit_rate)
    
    trades_list = high_hit_rate + low_hit_rate
    
    high_list = high_hit_rate
    low_list = low_hit_rate
    
    high_hit_rate = (len([x for x in high_hit_rate if x > 0]) / len(high_hit_rate)) * 100
    low_hit_rate = (len([x for x in low_hit_rate if x > 0]) / len(low_hit_rate)) * 100
    

    trades_dic = {**high_dic, **low_dic}
    #trades_dic = sorted(trades_dic.keys())
    total = leverage
    tracker = []
    for key, value in sorted(trades_dic.items()):
        total += trades_dic[key]
        tracker.append(total)
        
    trades = pd.DataFrame({'Date': list(trades_dic.keys()), 'Profit':list(trades_dic.values()) })
    growth_tracker = pd.DataFrame({'Date': sorted(list(trades_dic.keys())), 'Cumulative Value': tracker})
    growth_tracker = growth_tracker.set_index('Date')
    return  growth_tracker
    



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
    us = pd.read_csv("../RQ1_Finding_SimStock/main_result_ts2vec/{}.csv".format(data_name)) #nasdaq_nasdaq_2022
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

def finding_corr_list(index_name, data, num, same_ex, filter_):
    data.index = data.columns
    
    if same_ex == True:
        return list(data[index_name].sort_values(ascending=False).nlargest(num+1).index)
    else:
        return [index_name]+list(data[index_name].sort_values(ascending =False).filter(like=filter_).nlargest(num).index) # .T, .SS, .SZ
        
def finding_ts2vec_list(query, ts2vec, num, same_ex, filter_):
    if same_ex == True:
        return list(ts2vec.corr()[query].nlargest(num+1).index )
    else:
        return [query]+list(ts2vec[query].sort_values(ascending =False).filter(like=filter_).nlargest(num).index) # .T, .SS, .SZ


def finding_ss_list(query, ss, num, same_ex, filter_):
    if same_ex == True:
        return list(ss.corr()[query].nlargest(num+1).index )
    else:
        return [query]+list(ss[query].sort_values(ascending =False).filter(like=filter_).nlargest(num).index) # .T, .SS, .SZ
        

def get_index_and_col(result, data_df):
    result = pd.DataFrame(result)
    result.columns = data_df.columns
    result.index = data_df.columns
    return result

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