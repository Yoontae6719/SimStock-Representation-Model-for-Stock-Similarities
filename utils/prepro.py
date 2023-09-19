import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

def preprocess_stock(df, is_daily = True):
    '''
    Preprocess stock dataframe by generating rolling metrics for Open, High, Low, Close, and Volume columns.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe containing stock data with columns: Open, High, Low, Close, Volume.
    - is_daily (bool): Flag to determine if the stock data is daily. If set to False, it assumes hourly data.
    
    Returns:
    - df_pre (pd.DataFrame): Preprocessed dataframe with new rolling metrics columns added.
    '''
    
    if is_daily:
        rolling_size = [5, 10, 15, 20, 25]
    else:
        rolling_size = [6, 12, 18, 24, 30]
    
    df["Open06"] =  df['Open'].rolling(rolling_size[0]).sum() /   (rolling_size[0] * df['Open']) - 1
    df["Open12"] =  df['Open'].rolling(rolling_size[1]).sum() /  (rolling_size[1] * df['Open']) - 1
    df["Open18"] =  df['Open'].rolling(rolling_size[2]).sum() /  (rolling_size[2] * df['Open']) - 1
    df["Open24"] =  df['Open'].rolling(rolling_size[3]).sum() /  (rolling_size[3] * df['Open']) - 1
    df["Open30"] =  df['Open'].rolling(rolling_size[4]).sum() /  (rolling_size[4] * df['Open']) - 1

    df["High06"] =  df['High'].rolling(rolling_size[0]).sum() /   (rolling_size[0] * df['High']) - 1
    df["High12"] =  df['High'].rolling(rolling_size[1]).sum() /   (rolling_size[1] * df['High']) - 1
    df["High18"] =  df['High'].rolling(rolling_size[2]).sum() /   (rolling_size[2] * df['High']) - 1
    df["High24"] =  df['High'].rolling(rolling_size[3]).sum() /   (rolling_size[3] * df['High']) - 1
    df["High30"] =  df['High'].rolling(rolling_size[4]).sum() /   (rolling_size[4] * df['High']) - 1
    
    df["Low06"] =  df['Low'].rolling(rolling_size[0]).sum() /   (rolling_size[0] * df['Low']) - 1
    df["Low12"] =  df['Low'].rolling(rolling_size[1]).sum() /   (rolling_size[1] * df['Low']) - 1
    df["Low18"] =  df['Low'].rolling(rolling_size[2]).sum() /   (rolling_size[2] * df['Low']) - 1
    df["Low24"] =  df['Low'].rolling(rolling_size[3]).sum() /   (rolling_size[3] * df['Low']) - 1
    df["Low30"] =  df['Low'].rolling(rolling_size[4]).sum() /   (rolling_size[4] * df['Low']) - 1
    
    df["Close06"] =  df['Close'].rolling(rolling_size[0]).sum() /  (rolling_size[0] * df['Close']) - 1
    df["Close12"] =  df['Close'].rolling(rolling_size[1]).sum() /  (rolling_size[1] * df['Close']) - 1
    df["Close18"] =  df['Close'].rolling(rolling_size[2]).sum() /  (rolling_size[2] * df['Close']) - 1
    df["Close24"] =  df['Close'].rolling(rolling_size[3]).sum() /  (rolling_size[3] * df['Close']) - 1
    df["Close30"] =  df['Close'].rolling(rolling_size[4]).sum() /  (rolling_size[4] * df['Close']) - 1
    
    df["Volume06"] =  df['Volume'].rolling(rolling_size[0]).sum() /   (rolling_size[0] * df['Volume']) - 1
    df["Volume12"] =  df['Volume'].rolling(rolling_size[1]).sum() / (rolling_size[1] * df['Volume']) - 1
    df["Volume18"] =  df['Volume'].rolling(rolling_size[2]).sum() / (rolling_size[2] * df['Volume']) - 1
    df["Volume24"] =  df['Volume'].rolling(rolling_size[3]).sum() / (rolling_size[3] * df['Volume']) - 1
    df["Volume30"] =  df['Volume'].rolling(rolling_size[4]).sum() / (rolling_size[4] * df['Volume']) - 1

    df_pre = df.dropna()
    df_pre = df_pre.replace(np.inf, np.nan).dropna().reset_index(drop = True)
    df_pre = df_pre.replace(-np.inf, np.nan).dropna().reset_index(drop = True)
    
    return df_pre.reset_index(drop = True)




class DomainDataset(Dataset):
    """ Customized dataset for each domain"""
    def __init__(self,X, S):
        self.X = X                           # set data
        self.S = S                           # set Stock Sector

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.S[idx]]    # return list of batch data [data, labels]
    
    

def dataset_for_modeling(args, train_type = False):
    data = pd.read_csv("./data/{}.csv".format(args.train_dataset))
    test = pd.read_csv("./data/{}.csv".format(args.test_dataset))
    dataloaders = []
    input_var = ['Open06', 'Open12', 'Open18', 'Open24', 'Open30',
           'High06', 'High12', 'High18', 'High24', 'High30', 'Low06', 'Low12',
           'Low18', 'Low24', 'Low30', 'Close06', 'Close12', 'Close18', 'Close24',
           'Close30', 'Volume06', 'Volume12', 'Volume18', 'Volume24', 'Volume30']
    sector_var = ["IndustryCode_"]

        
    if not train_type:
        index_range = [126, 126*2, 126*3, 126*4, 126*5, 126*6, 126*7, 126*8, 126*9, 126*10]
        data_0 = data[data["Date_index"] <= index_range[0]]
        data_1 = data[(data["Date_index"] > index_range[0]) & (data["Date_index"] <= index_range[1]) ]
        data_2 = data[(data["Date_index"] > index_range[1]) & (data["Date_index"] <= index_range[2]) ]
        data_3 = data[(data["Date_index"] > index_range[2]) & (data["Date_index"] <= index_range[3]) ]
        data_4 = data[(data["Date_index"] > index_range[3]) & (data["Date_index"] <= index_range[4]) ]
        data_5 = data[(data["Date_index"] > index_range[4]) & (data["Date_index"] <= index_range[5]) ]
        data_6 = data[(data["Date_index"] > index_range[5]) & (data["Date_index"] <= index_range[6]) ]
        data_7 = data[(data["Date_index"] > index_range[6]) & (data["Date_index"] <= index_range[7]) ]
        data_8 = data[(data["Date_index"] > index_range[7]) & (data["Date_index"] <= index_range[8]) ]
        data_9 = data[(data["Date_index"] > index_range[8]) & (data["Date_index"] <= index_range[9]) ]
        data_a = data[(data["Date_index"] > index_range[9])]


        data_0_in= data_0[input_var].values
        data_1_in= data_1[input_var].values
        data_2_in= data_2[input_var].values
        data_3_in= data_3[input_var].values
        data_4_in= data_4[input_var].values
        data_5_in= data_5[input_var].values
        data_6_in= data_6[input_var].values
        data_7_in= data_7[input_var].values
        data_8_in= data_8[input_var].values
        data_9_in= data_9[input_var].values
        data_a_in= data_a[input_var].values

        data_0_se= data_0[sector_var].values
        data_1_se= data_1[sector_var].values
        data_2_se= data_2[sector_var].values
        data_3_se= data_3[sector_var].values
        data_4_se= data_4[sector_var].values
        data_5_se= data_5[sector_var].values
        data_6_se= data_6[sector_var].values
        data_7_se= data_7[sector_var].values
        data_8_se= data_8[sector_var].values
        data_9_se= data_9[sector_var].values
        data_a_se= data_a[sector_var].values

        data_in = [data_0_in
                  ,data_1_in
                  ,data_2_in
                  ,data_3_in
                  ,data_4_in
                  ,data_5_in
                  ,data_6_in
                  ,data_7_in
                  ,data_8_in
                  ,data_9_in
                  ,data_a_in]

        data_se = [data_0_se
                  ,data_1_se
                  ,data_2_se
                  ,data_3_se
                  ,data_4_se
                  ,data_5_se
                  ,data_6_se
                  ,data_7_se
                  ,data_8_se
                  ,data_9_se
                  ,data_a_se]

        dataloaders=[]
        for i in range(len(data_in)-1):
            temp_X = data_in[i]
            temp_S = data_se[i]
                        
            domain_dataset = DomainDataset(temp_X, temp_S) # create dataset for each domain
            temp_dataloader = DataLoader(domain_dataset, batch_size= args.batch_size, shuffle=True, num_workers=0)
            dataloaders.append(temp_dataloader)
            
        return dataloaders
    else:
        
        data_test_in= test[input_var].values
        data_test_se= test[sector_var].values
        domain_data_testset = DomainDataset(data_test_in, data_test_se) 
        test_dataloader = DataLoader(domain_data_testset, batch_size= args.batch_size, shuffle=False, num_workers=0)
    
    return test_dataloader