#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt


# In[21]:


DATA_DIR = 'data'


# In[22]:


def prepare_data(time_frame):
  btc_file_name = DATA_DIR+'/BTC_2019_2023_'+time_frame+'.csv'
  eth_file_name = DATA_DIR+'/ETHUSDT_'+time_frame+'.csv'

  btc_df = pd.read_csv(btc_file_name)
  eth_df = pd.read_csv(eth_file_name)

  btc_df = btc_df.set_index('datetime')
  eth_df = eth_df.set_index('datetime')

  common_datetime = btc_df.index.intersection(eth_df.index)
  btc_df_common = pd.DataFrame(btc_df.loc[common_datetime]['close']).rename(columns={'close':'btc_close'})
  eth_df_common = pd.DataFrame(eth_df.loc[common_datetime]['close']).rename(columns={'close':'eth_close'})

  data = pd.concat([btc_df_common,eth_df_common],axis=1)
  data['btc/eth'] = data['btc_close']/data['eth_close']

  data['spread_avg'] = data['btc/eth'].rolling(window=30).mean()
  data['spread_std'] = data['btc/eth'].rolling(window=30).std()
  data['upper_band'] = data['spread_avg'] + 2*data['spread_std']
  data['lower_band'] = data['spread_avg'] - 1*data['spread_std']

  return data


# In[ ]:


def pair_trading_strategy_bb(data,initial_balance=1):
  balance_eth = initial_balance
  btc_holdings=0
  in_btc = False
  trade_log = []

  for i in range(len(data)):
    date = data.index[i]
    ratio = data['btc/eth'].iloc[i]
    upper_band = data['upper_band'].iloc[i]
    lower_band = data['lower_band'].iloc[i]

    if ratio < lower_band and not in_btc:
      btc_holdings = balance_eth / ratio
      in_btc = True
      trade_log.append({
        'date':date,
        'action':'buy_btc',
        'ratio':ratio,
        'btc_holdings':btc_holdings,
        'eth_balance':balance_eth,
      })

    elif ratio > upper_band and in_btc:
      balance_eth = btc_holdings * ratio
      btc_holdings = 0
      in_btc = False
      trade_log.append({
        'date':date,
        'action':'sell_btc',
        'ratio':ratio,
        'btc_holdings':btc_holdings,
        'eth_balance':balance_eth,
      })

  if in_btc:
    final_ratio = data['btc/eth'].iloc[-1]
    balance_eth = btc_holdings * final_ratio
    trade_log.append({
      'date':data.index[-1],
      'action':'mark_to_market',
      'ratio':final_ratio,
      'btc_holdings':0,
      'eth_balance':balance_eth,
    })
    
    trade_log = pd.DataFrame(trade_log)
    trade_log.to_csv('BTC_ETH_PAIRS_TRADING_LONG_ONLY.csv')
    return balance_eth,trade_log


# In[16]:


def calculate_correlation(data, btc_column='btc_close', eth_column='eth_close'):
    """
    Calculate the correlation between BTC-USDT and ETH-USDT prices.

    Parameters:
        data (pd.DataFrame): A DataFrame containing BTC and ETH price columns.
        btc_column (str): The name of the column containing BTC prices.
        eth_column (str): The name of the column containing ETH prices.

    Returns:
        float: The correlation coefficient between BTC and ETH prices.
    """
    # Check if the required columns exist in the DataFrame
    if btc_column not in data.columns or eth_column not in data.columns:
        raise ValueError(f"DataFrame must contain columns '{btc_column}' and '{eth_column}'.")

    # Drop rows with missing values to avoid issues during correlation calculation
    clean_data = data[[btc_column, eth_column]].dropna()

    # Calculate the correlation coefficient
    correlation = np.log10(clean_data[btc_column]).corr(np.log10(clean_data[eth_column]))
    
    return correlation


# In[31]:


def pairs_trading_strategy_zscore(data,entry_threshold=2,exit_threshold=0.5):

    # Log-transform the prices
    data['log_btc'] = np.log(data['btc_close'])
    data['log_eth'] = np.log(data['eth_close'])

    # Perform cointegration test
    score, p_value, _ = coint(data['log_btc'], data['log_eth'])
    print(f"Cointegration Test Result:\nScore: {score}\nP-value: {p_value}")

    if p_value < 0.05:
        print("The series are cointegrated, proceeding with the strategy.")
    else:
        print("The series are not cointegrated. Strategy may not be effective.")
        
    # Calculate hedge ratio using OLS regression
    ols_result = sm.OLS(data['log_btc'], sm.add_constant(data['log_eth'])).fit()
    hedge_ratio = ols_result.params[1]
    print(f"Hedge Ratio: {hedge_ratio}")

    # Calculate the spread
    data['spread'] = data['log_btc'] - hedge_ratio * data['log_eth']

    # Calculate z-score of the spread
    data['z_score'] = (data['spread'] - data['spread'].mean()) / data['spread'].std()

    # Generate signals
    data['long_signal'] = data['z_score'] < -entry_threshold
    data['short_signal'] = data['z_score'] > entry_threshold
    data['exit_signal'] = (data['z_score'] > -exit_threshold) & (data['z_score'] < exit_threshold)

    # Simulate positions
    data['position'] = 0
    data.loc[data['long_signal'], 'position'] = 1  # Long BTC, Short ETH
    data.loc[data['short_signal'], 'position'] = -1  # Short BTC, Long ETH
    data['position'] = data['position'].replace(to_replace=0, method='ffill')  # Hold position
    data.loc[data['exit_signal'], 'position'] = 0  # Exit positions

    # Backtest returns
    data['btc_return'] = data['btc_close'].pct_change()
    data['eth_return'] = data['eth_close'].pct_change()
    data['strategy_return'] = data['position'].shift(1) * (data['btc_return'] - hedge_ratio * data['eth_return'])

    # Calculate cumulative returns
    data['cumulative_strategy_return'] = (1 + data['strategy_return']).cumprod()
    print(data['cumulative_strategy_return'])

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(data['cumulative_strategy_return'], label='Strategy Return')
    plt.title('Cumulative Strategy Return')
    plt.legend()
    plt.show()

    # Calculate drawdowns
    data['rolling_max'] = data['cumulative_strategy_return'].cummax()
    data['drawdown'] = data['cumulative_strategy_return'] / data['rolling_max'] - 1
    max_drawdown = data['drawdown'].min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Calculate Sharpe ratio
    # Assume risk-free rate is 0 for simplicity
    risk_free_rate = 0
    strategy_annual_return = data['strategy_return'].mean() * 252
    strategy_annual_volatility = data['strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = (strategy_annual_return - risk_free_rate) / strategy_annual_volatility
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


# In[32]:


data = prepare_data('4h')
pairs_trading_strategy_zscore(data)
    


# In[ ]:




