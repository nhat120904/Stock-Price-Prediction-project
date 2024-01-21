## Importing necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from functools import partial
import plotly.graph_objects as go
from datetime import datetime
import plotly.subplots as ms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from technical_generator import TechnicalGenerator

stock_df = pd.read_csv('Apple.csv',header = None)
stock_df.columns = ['Date','Open','High','Low','Close','Volume']
# Grouping the `stock_df` DataFrame by the 'Date' column and then aggregating the data for
# each unique date.
stock_df = stock_df.groupby('Date').agg({
        'Close' : 'last',
        'Open' : 'first',
        'High' : 'max',
        'Low' : 'min',
        'Volume' : 'sum'
}).reset_index()

def transform_data(df):
    df['Stochastic Oscillator'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['Absolute Returns'] = df['Close'] - df['Close'].shift(1)
    df['Close Normalized'] = df['Close'] / df['Close'].shift(1)
    df['Open Normalized'] = df['Open'] / df['Close'].shift(1)
    df['High Value Normalized'] = df['High'] / df['Close'].shift(1)
    df['Low Value Normalized'] = df['Low'] / df['Close'].shift(1)
    df['Volume Normalized'] = df['Volume'] / df['Volume'].shift(1).rolling(window=5).mean()
    df['Volatility'] = df['Volume'].rolling(window=9).var() / 1e16
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_macd(df, period1=9, period2=5):
    alpha1 = 2 / (period1 + 1)
    alpha2 = 2 / (period2 + 1)
    EMA9day = df['Close'].ewm(span=period1, adjust=False).mean()
    EMA5day = df['Close'].ewm(span=period2, adjust=False).mean()
    df['MACD'] = EMA5day - EMA9day
    return df

def calculate_gains_losses(df):
    df['Gains'] = (df['Close'] - df['Open']) * 100 / df['Open']
    df['Gains'] = df['Gains'].apply(lambda x: max(0, x))
    df['Losses'] = (df['Open'] - df['Close']) * 100 / df['Open']
    df['Losses'] = df['Losses'].apply(lambda x: max(0, x))
    return df

def compute_rsi(df):
    rsi_data = []
    eps = 1e-8
    for i in range(9, len(df)):
        rsi_gains = np.array(df.iloc[i-9:i]['Gains'])
        average_gains = np.mean(rsi_gains[rsi_gains > 0])
        average_gains = 0 if np.isnan(average_gains) else average_gains
        rsi_losses = np.array(df.iloc[i-9:i]['Losses'])
        average_losses = np.mean(rsi_losses[rsi_losses > 0])
        average_losses = 0 if np.isnan(average_losses) else average_losses
        den = 1 + average_gains / (average_losses + eps)
        rsi_data.append(100 - (100 / den))
    
    df = df.iloc[9:]
    df['RSI'] = rsi_data
    return df

def technical_generator(df):
    df = transform_data(df)
    df = calculate_macd(df)
    df = calculate_gains_losses(df)
    df = compute_rsi(df)
    df['Sine'] = np.sin(2 * np.pi / 20 * pd.DatetimeIndex(data=df['Date'], yearfirst=True).day)
    df['Cosine'] = np.cos(2 * np.pi / 20 * pd.DatetimeIndex(data=df['Date'], yearfirst=True).day)
    return df

class CreateInOutSequence(Dataset):
    def __init__(self,data,sequence_length,prediction_length):
        ## Initializing the dataframe and the window length i.e. number of previous days which will be used at a time to predict
        ## today's Close
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
    def __len__(self):
        ## As it picks indices randomly from [0,len], we keep len =  len(df) - seq_len which denotes the last index which can be
        ## used to create a batch as we need seq_len rows ahead of it
        return len(self.data) - self.sequence_length - self.prediction_length
        
    def __getitem__(self,index):
        ## Slicing the dataframe from input index to input index + seq_len to get the input data 
        input_data = self.data[index : index + self.sequence_length]
        input_list = input_data.values.tolist()
        input = torch.Tensor(input_list)
        
        ## Returning the Closes of next day as the output for each day in the input
        ## Converting both the input and output to tensors before returning
        output = self.data.loc[index + self.sequence_length : index + self.sequence_length + self.prediction_length-1, 'Close Normalized'].values.tolist()
        output = torch.Tensor(output)
        
        return input,output
    
stock_df = technical_generator(stock_df)
input_features = ['Close Normalized', 'Open Normalized','High Value Normalized','Low Value Normalized','Volume Normalized',
                'Stochastic Oscillator', 'Absolute Returns','Volatility','MACD','RSI','Sine','Cosine']
df_final = stock_df[input_features].copy()
df_final.to_csv('final_data.csv',index = False)

# sequence_length = 12
# prediction_length = 5
# sequenced_data = CreateInOutSequence(df_final,sequence_length,prediction_length)

# def train_test_splitter(dataset, split = 0.8):
#     indices = list(range(len(dataset)))

#     #splitting the indices according to the decided split 
#     train_indices, test_indices = train_test_split(indices, train_size=split, shuffle=False)
#     val_indices, test_indices = train_test_split(test_indices, train_size=0.5, shuffle=False)

#     # Create the training , validation and test datasets
#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     val_dataset= torch.utils.data.Subset(dataset, val_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)
#     train_size=len(train_dataset)
#     test_size=len(val_dataset)
#     val_size=len(test_dataset)
#     return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size



