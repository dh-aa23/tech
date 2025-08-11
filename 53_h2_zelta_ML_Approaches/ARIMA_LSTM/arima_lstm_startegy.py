import time
import pandas as pd
import pandas_ta as ta
import talib as t
from untrade.client import Client
from pprint import pprint
from tqdm import tqdm
import talib as t
import warnings
import numpy as np
warnings.filterwarnings('ignore')


def process_data(df):
    return df

def strat(df):
    df['Signal'] = np.where((df['Signal'] == -1) | (df['Signal'] == -2), 0, df['Signal'])
    df['Signal_Long'] = (df['Signal']==1) 
    df['Signal_Short'] =(df['Signal']==2) 
    df['signals'] = np.where(df['Signal_Long'], 1, np.where(df['Signal_Short'], 2, 0))

    df["trade_type"] = "Hold"
    in_trade_long, in_trade_short = False, False
    df['remarks'] = 'Algo_Exit'
    for i in tqdm(range(len(df))):
        if df['signals'].iloc[i - 1] == -1:
            df.at[i, 'signals'] = 2
            df['trade_type'].iloc[i]='Short_entry'
            in_trade_short = True
        if df['signals'].iloc[i - 1] == -2:
            df.at[i, 'signals'] = 1
            df['trade_type'].iloc[i]='Long_entry'
            in_trade_long = True
        elif df['signals'].iloc[i] == 2 and in_trade_long:
            df.at[i, 'signals'] = -1
            df['trade_type'].iloc[i]='long_square_off'
            in_trade_long = False
        elif df['signals'].iloc[i] == 1 and in_trade_short:
            df.at[i, 'signals'] = -2
            df['trade_type'].iloc[i]='short_square_off'
            in_trade_short = False
        elif df['signals'].iloc[i] == 2 and not in_trade_short:
            in_trade_short = True
            df['trade_type'].iloc[i]='Long_entry'
            in_trade_long = False
        elif df['signals'].iloc[i] == 1 and not in_trade_long:
            in_trade_long = True
            df['trade_type'].iloc[i]='Short_entry'
            in_trade_short = False
    
    df_1 = df
    long_open = False
    short_open = False

    for index, row in df_1.iterrows():
        signal = row['signals']
        if signal == 2 and short_open:
            df_1.at[index, 'signals'] = 0
            df["trade_type"].iloc[i] = "Hold"
        elif signal == 1 and long_open:
            df["trade_type"].iloc[i] = "Hold"
            df_1.at[index, 'signals'] = 0
        elif signal == 1 and long_open != True:
            long_open = True
            df["trade_type"].iloc[i] = "Long_Entry"
        elif signal == 2 and short_open != True:
            short_open = True
            df["trade_type"].iloc[i] = "Short_Entry"
        elif signal == -1:
            if long_open:
                long_open = False
            else:
                df_1.at[index, 'signals'] = 0
                df["trade_type"].iloc[i] = "Hold"
        elif signal == -2:
            if short_open:
                short_open = False
            else:
                df_1.at[index, 'signals'] = 0
                df["trade_type"].iloc[i] = "Hold"


    for i in tqdm(range(len(df))):
        if df['signals'].iloc[i] == 2 :
            df['signals'].iloc[i] = -1
        elif df['signals'].iloc[i] == -2 :
            df['signals'].iloc[i] = 1
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
    df_signals = pd.read_csv("signals arima n lstm.csv")
    df_signals = df_signals.rename(columns={'datetime': 'timestamp'})
    
    df_full = pd.read_csv("btcusdt_1h.csv")  
    df_full = df_full.rename(columns={'datetime': 'timestamp'})


    df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

    # Merge the dataframes on the 'timestamp' column (assuming 'datetime' exists in both dataframes)
    data = pd.merge(df_signals, df_full[['timestamp', 'open', 'high', 'low', 'close']], on='timestamp', how='left')
    
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