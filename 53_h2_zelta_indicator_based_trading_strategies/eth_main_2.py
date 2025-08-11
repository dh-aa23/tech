
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import talib as t
import pandas_ta as ta
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from untrade.client import Client
warnings.filterwarnings("ignore")


def process_data(df):

    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_Open'] = (df['open'] + df['close']) / 2
    df['HA_Open'] = df['HA_Open'].shift(1)
    df['HA_Open'].iloc[0] = df['open'].iloc[0]
    df['HA_High'] = df[['HA_Open', 'HA_Close']].join(df['high']).max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close']].join(df['low']).min(axis=1)

    df['ATR'] = t.ATR(df.HA_High, df.HA_Low, df.HA_Close, timeperiod=14)
    df['adx'] = t.ADX(df['HA_High'], df['HA_Low'], df['HA_Close'], timeperiod=14)
    df['+di'] = t.PLUS_DI(df['HA_High'], df['HA_Low'], df['HA_Close'], timeperiod=15)
    df['-di'] = t.MINUS_DI(df['HA_High'], df['HA_Low'], df['HA_Close'], timeperiod=15)
    df['rsi'] = t.RSI(df['HA_Close'], timeperiod=30)
    df['vwap'] = t.SUM(df['volume'] * (df['high'] + df['low'] + df['close']) / 3, timeperiod=14) / t.SUM(df['volume'], timeperiod=14)

    df['EMA']=t.EMA(df['HA_Close'],timeperiod=20)
    df['UB'] = df['EMA'] + 2 * df['ATR']
    df['LB'] = df['EMA'] - 2 * df['ATR']
    df['width'] = df['UB'] - df['LB']
    df['vol_region'] = np.where(df['width'] > df['width'].mean(),'High','Low')

    df['ema_fast'] = t.EMA(df['volume'], timeperiod=20)
    df['ema_slow'] = t.EMA(df['volume'], timeperiod=50)
    df["ema_fastest"] = t.EMA(df["close"], timeperiod=7)
    df['vol_fast'] = t.EMA(df['HA_Close'],timeperiod = 20)
    df['vol_slow'] = t.EMA(df['HA_Close'],timeperiod = 50)
    df['ADX'] = t.ADX(df['high'],df['low'],df['close'],timeperiod = 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = t.MACD(df['HA_Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_Z_Score'] = (df['MACD_hist'] - df['MACD_hist'].mean())/df['MACD_hist'].std()

    df["spike"] = df["HA_Close"] - df["HA_Open"]
    df["vol_index"] = df["spike"].rolling(window=20).std()
    df["vol_index_ma"] = df["vol_index"].rolling(window=20).mean()
    df["LL"] = df["HA_Low"].rolling(window=20).min()
    df["HH"] = df["HA_High"].rolling(window=20).max()

    df.ta.bbands(close='close', length=20, std=2.5, append=True)
    df['EMA_200'] = t.EMA(df['HA_Close'], timeperiod=200)
    df['EMA_10'] = t.EMA(df['HA_Close'], timeperiod=10)
    df['EMA_20'] = t.EMA(df['HA_Close'], timeperiod=20)
    df['EMA_50'] = t.EMA(df['HA_Close'], timeperiod=50)
    df['EMA_100']= t.EMA(df['HA_Close'], timeperiod =100)
    df['SAR'] = t.SAR(df['HA_High'], df['HA_Low'], acceleration=0.02, maximum=0.5)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = t.MACD(df['HA_Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df["close_std"] = df["close"].rolling(window=20).std()
    df["rolling_mean_std"] = df["close_std"].rolling(window=20).mean()
    df["std_check"] = df["close_std"] < df["rolling_mean_std"]

    return df

def strat(df):

    df['signals'] = 0
    df['trade_type'] = 'hold'
    in_trade_long = False
    in_trade_short = False
    entry_price_long = 0
    entry_price_short = 0

    for i in tqdm(range(len(df))):
        curr_row = df.iloc[i, :]
        prev_row = df.iloc[i - 1, :]
        if not in_trade_long and not in_trade_short:
            if  (curr_row['HA_Close'] > curr_row['EMA_200'] and df['close_std'].iloc[i] < df['rolling_mean_std'].iloc[i] and
                    prev_row['MACD'] < prev_row['MACD_signal'] and curr_row['MACD'] > curr_row['MACD_signal'] 
                    and curr_row['EMA_50'] > curr_row['EMA_200'] and prev_row['EMA_50'] > prev_row['EMA_200'] ):
                df.at[i, 'signals'] = 1
                df.at[i,'trade_type'] = 'Long_entry'# Long signal
                sl = df['LL'].iloc[i] - 0.08 * df['LL'].iloc[i-1]
                # sl = df['SL'].iloc[i]
                entry_time = df['datetime'].iloc[i]
                in_trade_long = True
            elif (curr_row['HA_Close'] < curr_row['EMA_200'] and df['close_std'].iloc[i] < df['rolling_mean_std'].iloc[i] and
                    prev_row['MACD'] > prev_row['MACD_signal'] and curr_row['MACD'] < curr_row['MACD_signal']  
                    and curr_row['EMA_50'] < curr_row['EMA_200'] and prev_row['EMA_50'] < prev_row['EMA_200']): 
                df.at[i, 'signals'] = -1  # Short signal
                df.at[i,'trade_type'] = 'Short_entry'
                atr_value = df['ATR'].iloc[i]
                sl = df['HH'].iloc[i] + 0.08 * df['HH'].iloc[i-1]
                # sl = df['SL'].iloc[i]
                # sl = 1 if df['vol_region'].iloc[i] == 'High' else 2
                in_trade_short = True
        
        elif in_trade_long:
            if ((curr_row['rsi'] < 20) or (curr_row['low'] < sl < curr_row['high']) or
               (df['vol_index'].iloc[i] > df['vol_index_ma'].iloc[i] and df['rsi'].iloc[i] < 25) or
               ()):
                df.at[i, 'signals'] = -1  # Exit long position
                df.at[i,'trade_type'] = 'Long_entry'
                in_trade_long = False
                sl = 0
                entry_price_long = 0
           
        
        elif in_trade_short:
            if ((curr_row['rsi'] > 80) or (curr_row['low'] < sl < curr_row['high']) or
               (df['vwap'].iloc[i] > df['ema_fastest'].iloc[i] and df['rsi'].iloc[i] > 70)
               ):
                df.at[i, 'signals'] = 1  # Exit short position
                df.at[i,'trade_type'] = 'Short_entry'
                in_trade_short = False
                sl = 0
                entry_price_short = 0

    return df
def perform_backtest(csv_file_path):

    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="test",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,# Adjust leverage as needed
    )
    return result
def main():
     data = pd.read_csv("eth_30_min_2020_23.csv")
     
     processed_data = process_data(data)
     

     result_data = strat(processed_data)
     
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



