import time
import numpy as np
import pandas as pd
import talib as ta
from tqdm import tqdm
from untrade.client import Client
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import talib as t
import warnings
warnings.filterwarnings('ignore')


def process_data(df):

    
    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_Open'] = df['open']
    for i in range(1, len(df)):
        df.at[i, 'HA_Open'] = (df.at[i-1, 'HA_Open'] + df.at[i-1, 'HA_Close']) / 2
    df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)

    df["vol_sma_fast"] = df["volume"].rolling(window=20).mean()
    df["vol_sma_slow"] = df["volume"].rolling(window=50).mean()
    df["rsi"] = ta.RSI(df["HA_Close"], timeperiod=20)
    df["rsi_smooth"] = ta.EMA(df["rsi"], timeperiod=14)
    df['vwap'] = ta.SUM(df['volume'] * (df['high'] + df['low'] + df['HA_Close']) / 3, timeperiod=14) / ta.SUM(df['volume'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['HA_Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['adx'] = ta.ADX(df['HA_High'], df['HA_Low'], df['HA_Close'], timeperiod=14)
    
    df["ema_fast"] = ta.EMA(df["HA_Close"], timeperiod=20)
    df['ATR'] = ta.ATR(df['HA_High'], df['HA_Low'], df['HA_Close'],timeperiod = 14)
    df["ema_slow"] = ta.EMA(df["HA_Close"], timeperiod=50)
    df["ema_fastest"] = ta.EMA(df["HA_Close"], timeperiod=5)

    df['bb_middle'] = ta.SMA(df['HA_Close'], timeperiod=10)
    df['bb_upper'] = df['bb_middle'] + 1.50 * df['HA_Close'].rolling(window=10).std()
    df['bb_lower'] = df['bb_middle'] - 1.50 * df['HA_Close'].rolling(window=10).std()
    
    df["spike"] = df["HA_Close"] - df["HA_Open"]
    df["vol_index"] = df["spike"].rolling(window=20).std()
    df["vol_index_ma"] = df["vol_index"].rolling(window=20).mean()
    
    df["LL"] = df["HA_Low"].rolling(window=20).min()
    df["HH"] = df["HA_High"].rolling(window=20).max()

    return df

def strat(df):
    
    df["signal"] = 0
    position = None
    entry_time = None
    df["trade_type"] = "Hold"
    df["sl"] = 0
    sl = 0
    sl_short = 0
    
    for i in tqdm(range(1, len(df))):
        curr_row = df.iloc[i, :]
        prev_row = df.iloc[i - 1, :]
        # long_entry
        if (curr_row["rsi_smooth"] > 60) and (
           curr_row["vol_sma_fast"] > curr_row["vol_sma_slow"] and (prev_row['HA_Close'] > prev_row['bb_upper'])
        ):
           if (
               (curr_row["ema_fast"] > curr_row["ema_slow"])
               and (curr_row["ema_fastest"] > curr_row["ema_fast"]) and (df['adx'].iloc[i]>20) 
           ):
               if not position:
                   df.at[i, "signal"] = 1
                   df.at[i, "trade_type"] = 'Long_entry'
        
        
                   df.at[i, "sl"] = prev_row['HA_Close'] - 0.05 * prev_row['LL']
                   sl = df.at[i,"sl"]
                   entry_time = curr_row["datetime"]
                   position = "long"
        # long_exit
        if (curr_row["rsi"] < 25) and (
            curr_row["vol_index"] < curr_row["vol_index_ma"]) :
           if position == "long":  # to_ask
               df.at[i, "signal"] = -1
               df.at[i, "trade_type"] = 'long_square_off'
               entry_time = None
               position = None
        
        
        if (position == "long") and (curr_row["HA_Low"]<sl<curr_row["HA_High"]):
            df.at[i, "signal"] = -1
            sl = 0
            df.at[i, "trade_type"] = 'long_square_off'
            entry_time = None
            position = None
        
        # short_entry
        if (curr_row["ema_fast"] < curr_row["ema_slow"]) and (curr_row['HA_Close'] > curr_row['bb_lower'])  and prev_row["rsi_smooth"] < 28:
          
           if (
               (
                   (prev_row["ema_fastest"] < prev_row["vwap"])
                   and (curr_row["ema_fastest"] > curr_row["vwap"])
               )
               and (curr_row["rsi_smooth"] < 27)
               
           ):
               if not position:
                   df.at[i, "signal"] = -1
                   df.at[i, "trade_type"] = 'short_entry'
                   df.at[i, "sl"] = curr_row["close"] + 0.01*curr_row["close"]
                   sl_short = df.at[i,"sl"]
                   entry_time = curr_row["datetime"] #set entry time 
                   position = "short"
        # short_exit
        if curr_row["vwap"] > curr_row["ema_fastest"] and curr_row["rsi"] > 72: 
           if (
               position == "short"
           ):
               
               df.at[i, "signal"] = 1 #square off short condition
               df.at[i, "trade_type"] = 'short_square_off'
               position = None
               
        if (position == "short") & (curr_row["HA_Low"] < sl_short < curr_row["HA_High"]):
            df.at[i, "signal"] =  1 #square off short condition
            sl_short = 0
            df.at[i, "trade_type"] = 'short_square_off'
            entry_time = None 
            position = None 

    
    df['signals'] = df['signal']
    return df

def perform_backtest(csv_file_path):
     client = Client()
     result = client.backtest(
         jupyter_id="test",  # the one you use to login to jupyter.untrade.io
         file_path=csv_file_path,
         leverage=1,  # Adjust leverage as needed
     )
     return result

def main():
     data = pd.read_csv("btcusdt_15m.csv")
     
     processed_data = process_data(data)
     

     result_data = strat(processed_data)
     # data = [data['signal']!=0]
     result_data = result_data.dropna()

     csv_file_path = "results.csv"

     result_data.to_csv(csv_file_path, index=False)

     backtest_result = perform_backtest(csv_file_path)
     # backtest_result = perform_backtest(csv_file_path)

     # No need to use following code if you are using perform_backtest_large_csv
     last_value = None
     for value in backtest_result:
        print(value)  # Uncomment to see the full backtest result (backtest_result is a generator object)
        last_value = value 
if __name__ == "__main__":
     main()  