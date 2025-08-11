import time
import pandas as pd
import talib as t
import pandas_ta as ta
from tqdm import tqdm
from untrade.client import Client
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import talib as t
import warnings
from pykalman import KalmanFilter
import numpy as np
from datetime import datetime,timedelta

import requests
from arch import arch_model
warnings.filterwarnings('ignore')


def process_data(df,look_back_days):
    
    def hawkes_process(data: pd.Series,kappa):
        alpha=np.exp(-kappa)
        arr=data.to_numpy()
        output=np.zeros(len(data))
        output[:]=np.nan
        for i in range(1,len(data)):
            if np.isnan(output[i-1]):
                output[i]=arr[i]
            else:
                output[i]=output[i-1]*alpha+arr[i]
        return pd.Series(output,index=data.index)

    def fib(entry, maximum):
        diff = maximum - entry
        level1 = maximum - 0.382 * diff
        level2 = maximum - 0.50 * diff
        level3 = maximum - 0.618 * diff
        return {'23.6' : level1 , '38.2' : level2 , '50.0':level3}


    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    df['HA_Open'] = (df['open'] + df['close']) / 2
    df['HA_Open'] = df['HA_Open'].shift(1)
    df['HA_Open'].iloc[0] = df['open'].iloc[0]
    df['HA_High'] = df[['HA_Open', 'HA_Close']].join(df['high']).max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close']].join(df['low']).min(axis=1)
    
    df['atr'] = ta.atr(np.log(df['HA_High']), np.log(df['HA_Low']), np.log(df['HA_Close']), window=look_back_days)
    df['ATR'] = ta.atr(df['HA_High'],df['HA_Low'],df['HA_Close'], window=look_back_days)
    df['norm_range']=(np.log(df['high'])-np.log(df['low']))/df['atr']
    
    df['hawkes']=hawkes_process(df['norm_range'],0.1)
    df['adx'] = t.ADX(df['high'], df['low'], df['close'], timeperiod=15)
    df['+di'] = t.PLUS_DI(df['HA_High'], df['HA_Low'], df['HA_Close'], timeperiod=15)
    df['-di'] = t.MINUS_DI(df['HA_High'], df['HA_Low'], df['HA_Close'], timeperiod=15)
    df['rsi'] = t.RSI(df['HA_Close'], timeperiod=14)
    df['hawkes_q95']=df['hawkes'].rolling(look_back_days).quantile(0.95)
    df['hawkes_q05']=df['hawkes'].rolling(look_back_days).quantile(0.05)
    macd, macd_signal, macd_hist = t.MACD(
        df['HA_Close'], 
        fastperiod=12,  # Short-term EMA
        slowperiod=26,  # Long-term EMA
        signalperiod=9  # Signal line EMA
    )
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_hist
    df['Z_Score'] = (df['MACD_Histogram'] - df['MACD_Histogram'].mean())/df['MACD_Histogram'].std()
    
    df['bb_sma']=df['HA_Close'].rolling(20).mean()
    df['bb_std']=df['HA_Close'].rolling(20).std()
    df['bb_up']=df['bb_sma']+2.5*df['bb_std']
    df['bb_down']=df['bb_sma']-2.5*df['bb_std']
    
    df["spike"] = df["HA_Close"] - df["HA_Open"]
    df["vol_index"] = df["spike"].rolling(window=20).std()
    df["vol_index_ma"] = df["vol_index"].rolling(window=20).mean()
    df['RSI_75'] = df['rsi'].rolling(window=252).apply(lambda x: np.percentile(x, 70), raw=False)
    df['RSI_30'] = df['rsi'].rolling(window=512).apply(lambda x: np.percentile(x, 25), raw=False)
    df['SAR'] = t.SAR(df['HA_High'], df['HA_Low'], acceleration=0.02, maximum=0.5)

    return df
def strat(df,adx_threshold):
    
    def hawkes_process(data: pd.Series,kappa):
        alpha=np.exp(-kappa)
        arr=data.to_numpy()
        output=np.zeros(len(data))
        output[:]=np.nan
        for i in range(1,len(data)):
            if np.isnan(output[i-1]):
                output[i]=arr[i]
            else:
                output[i]=output[i-1]*alpha+arr[i]
        return pd.Series(output,index=data.index)

    def fib(entry, maximum):
        diff = maximum - entry
        level1 = maximum - 0.382 * diff
        level2 = maximum - 0.50 * diff
        level3 = maximum - 0.618 * diff
        return {'23.6' : level1 , '38.2' : level2 , '50.0':level3}
    
    last_below=-1
    curr_sig=0
    df['signals']=0
    df['trade_type'] = 'hold'
    
    for i in tqdm(range(len(df))):
        if df['hawkes'].iloc[i]<df['hawkes_q05'].iloc[i]:
            last_below=i
            curr_sig=0
        if df['hawkes'].iloc[i]>df['hawkes_q95'].iloc[i] \
            and df['hawkes'].iloc[i-1]<=df['hawkes_q95'].iloc[i-1] \
            and last_below>0:
            change=df['close'].iloc[i]-df['close'].iloc[last_below]
            if change>0.0 and df['adx'].iloc[i]>adx_threshold and df['close'].iloc[i]>df['bb_up'].iloc[i] :
                curr_sig=1
                df['trade_type'].iloc[i] = 'long_entry'
            elif change<0.0 and df['adx'].iloc[i]>25 and df['close'].iloc[i]<df['bb_down'].iloc[i]:  
                curr_sig=2
                df['trade_type'] = 'short_entry'
        df['signals'].iloc[i]=curr_sig
    
    
    
    in_trade_long, in_trade_short = False, False
    
    
    for i in tqdm(range(len(df))):
        if df['signals'].iloc[i - 1] == -1:
            df.at[i, 'signals'] = 2
            in_trade_short = True
        if df['signals'].iloc[i - 1] == -2:
            df.at[i, 'signals'] = 1
            in_trade_long = True
        elif df['signals'].iloc[i] == 2 and in_trade_long:
            df.at[i, 'signals'] = -1
            in_trade_long = False
        elif df['signals'].iloc[i] == 1 and in_trade_short:
            df.at[i, 'signals'] = -2
            in_trade_short = False
        elif df['signals'].iloc[i] == 2 and not in_trade_short:
            in_trade_short = True
            in_trade_long = False
        elif df['signals'].iloc[i] == 1 and not in_trade_long:
            in_trade_long = True
            in_trade_short = False
    
    in_trade_long, in_trade_short = False, False
    maximum=0.0
    entry=0.0
    for i in tqdm(range(len(df))):
        if df['signals'].iloc[i]==-1 and in_trade_long:
            intrade_long=False
            entry=maximum=0.0
        elif df['signals'].iloc[i]==-2 and in_trade_short:
            intrade_short=False
            entry=maximum=0.0
        if in_trade_long:
            if df['HA_Close'].iloc[i]>maximum:
                maximum=df['HA_Close'].iloc[i]
            fibo=fib(entry,maximum)
            if ((df['HA_Close'].iloc[i]<fibo['23.6']) and (df['HA_Close'].iloc[i-1]>fibo['23.6']) and df['rsi'].iloc[i] > df['RSI_75'].iloc[i] and  df['SAR'].iloc[i] > df['HA_Close'].iloc[i])\
                or ((df['HA_Close'].iloc[i]<fibo['38.2']) and (df['HA_Close'].iloc[i-1]>fibo['38.2']) and df['rsi'].iloc[i] > df['RSI_75'].iloc[i] )\
                or ((df['HA_Close'].iloc[i]<fibo['50.0']) and (df['HA_Close'].iloc[i-1]>fibo['50.0'])) \
                or ((df['rsi'].iloc[i] < 20))\
                or ((df['vol_index'].iloc[i] > df['vol_index_ma'].iloc[i] and df['rsi'].iloc[i] < 30))\
                or ():
                df['signals'].iloc[i]=-1
                in_trade_long=False
                df['trade_type'].iloc[i]='long_square_off'
                entry=0.0
                maximum=0.0
        elif in_trade_short:
            if df['HA_Close'].iloc[i]<maximum:
                maximum=df['HA_Close'].iloc[i]
            fibo=fib(entry,maximum)
            if ((df['HA_Close'].iloc[i]>fibo['23.6']) and (df['HA_Close'].iloc[i-1]<fibo['23.6']) )\
                or ((df['HA_Close'].iloc[i]>fibo['38.2']) and (df['HA_Close'].iloc[i-1]<fibo['38.2']) )\
                or ((df['HA_Close'].iloc[i]>fibo['50.0']) and (df['HA_Close'].iloc[i-1]<fibo['50.0']) )\
                or ((df['rsi'].iloc[i] > 70))\
                or ((df['vol_index'].iloc[i] < df['vol_index_ma'].iloc[i] and df['rsi'].iloc[i] > 60)):
                df['signals'].iloc[i]=-2
                df['trade_type'].iloc[i]='short_square_off'
                in_trade_short=False
                entry=0.0
                maximum=0.0
            
    
        if df['signals'].iloc[i]==1 and not in_trade_long:
            in_trade_long=True
            entry=maximum=df['HA_Close'].iloc[i]
            in_trade_short=False
        elif df['signals'].iloc[i]==2 and not in_trade_short:
            in_trade_short=True
            entry=maximum=df['HA_Close'].iloc[i]
            in_trade_long=False
    
    
    df_1 = df
    long_open = False
    short_open = False
    
    for index, row in df_1.iterrows():
        signal = row['signals']
    
        if signal == 2 and short_open:
            df_1.at[index, 'signals'] = 0
        elif signal == 1 and long_open:
            df_1.at[index, 'signals'] = 0
        elif signal == 1 and long_open != True:
            long_open = True
        elif signal == 2 and short_open != True:
            short_open = True
    
        elif signal == -1:
            if long_open:
                long_open = False
            else:
                df_1.at[index, 'signals'] = 0
        elif signal == -2:
            if short_open:
                short_open = False
            else:
                df_1.at[index, 'signals'] = 0
    
    df_1['signal'] = df_1['signals']
    df_1.to_csv('signals.csv')
    # dg = df_1[df_1['signal'] != 0]
    dg = df
    dg.to_csv('signal_eth_hakwes.csv')

    for i in tqdm(range(len(dg))):
        if dg['signals'].iloc[i] == 2 :
            dg['signals'].iloc[i] = -1
        elif dg['signals'].iloc[i] == -2 :
            dg['signals'].iloc[i] = 1
    
    return dg

def perform_backtest(csv_file_path):
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="team53_zelta_hpps",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=2,# Adjust leverage as needed
    )
    return result


def main():
     data = pd.read_csv("ETHUSDT_4h.csv")
     
     processed_data = process_data(data,28)
     

     result_data = strat(processed_data,20)
    
     result_data = result_data.dropna()

     csv_file_path = "results.csv"

     result_data.to_csv(csv_file_path, index=False)

     backtest_result = perform_backtest(csv_file_path)
     
     last_value = None
     for value in backtest_result:
        print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
        last_value = value 
if __name__ == "__main__":
     main()  