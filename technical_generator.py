import numpy as np
import pandas as pd

class TechnicalGenerator:
    def __init__(self, data):
        self.data = data

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
    
    def compute_sin_cos(df):
        df['Sine'] = np.sin(2 * np.pi / 20 * pd.DatetimeIndex(data=df['Date'], yearfirst=True).day)
        df['Cosine'] = np.cos(2 * np.pi / 20 * pd.DatetimeIndex(data=df['Date'], yearfirst=True).day)
        return df

    def technical_generator(self): 
        self.data = self.transform_data(self.data)
        self.data = self.calculate_macd(self.data)
        self.data = self.calculate_gains_losses(self.data)
        self.data = self.compute_rsi(self.data)
        self.data = self.compute_sin_cos(self.data)
        return self.data