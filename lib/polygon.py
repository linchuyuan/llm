#!/usr/bin/env python3

from datetime import datetime, timedelta
import requests
import pdb
import pandas as pd
from datetime import time
import time as t
import os
import re

# api_key = 'gvhZSVGOB2_lwiCC71sngkBa8OtXKctd'
# Stock symbol (AAPL in this case)
# symbol = 'AAPL'
# Calculate the date range: from one year ago to today
# end_date = datetime.now()
# start_date = end_date - timedelta(days=360)

def parseOptionTicker(symbol):
    match = re.match(r"([A-Z]+)(\d{6})([CP])(\d{8})", symbol)
    if match:
        sym = match.group(1)
        exp = match.group(2)
        option_type = match.group(3)
        strike = float(match.group(4)) / 1000
        return sym, exp, option_type, float(strike)
    else:
        return None

def getApiKey(default=None):
    if 'api_key' in  os.environ:
        return os.environ['api_key']
    return default

def toFriday(date_obj):
    # Get the current day of the week (Monday = 0, Sunday = 6)
    current_day = date_obj.weekday()
    # Calculate the number of days to the closest Friday
    if current_day <= 4:
        # If the date is Monday to Friday, move forward to Friday
        days_to_friday = 4 - current_day
    else:
        # If it's Saturday or Sunday, move backwards to the previous Friday
        days_to_friday = - (current_day - 4)
    # Add the delta to the original date to get the closest Friday
    closest_friday = date_obj + timedelta(days=days_to_friday)
    return closest_friday

def getOptionTickers(stock_ticker, duration=timedelta(days=7)):
    api_key = getApiKey()
    now = datetime.now()
    start_date = toFriday(now - duration)
    end_date = toFriday(now + duration)
    url = (f"https://api.polygon.io/v3/reference/options/contracts"
           f"?underlying_ticker={stock_ticker}"
           f"&expiration_date.gte={start_date.strftime('%Y-%m-%d')}"
           f"&expiration_date.lte={end_date.strftime('%Y-%m-%d')}")

    params = {
        "adjusted": "true",   # Get adjusted stock prices
        "limit": 1000,       # Fetch as much data as possible
        "apiKey": api_key     # Polygon.io API key
    }
    option_tickers = []
    
    while url:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            contracts = data.get('results', [])
            status = data.get('status', None)
            err = data.get('error', None)
            if len(contracts) == 0 and status is not None and err is not None:
                print(data)
                t.sleep(70)
                continue
            
            # Append all tickers to the list
            option_tickers.extend([contract['ticker'] for contract in contracts])
            
            # Check for pagination
            url = data.get('next_url', None)
        else:
            print(f"Failed to retrieve data: {response.status_code}, {response.text}")
            return []

    return option_tickers
    
def getStockHistory(symbol, api_key=None, start_date=None, duration=None):
    if duration is None:
        duration = timedelta(days=14)
    api_key = getApiKey(api_key)
    end_date = None
    if start_date is None:
        end_date = datetime.now()
        start_date = end_date - duration
    else:
        end_date = start_date + duration

    # Initialize an empty DataFrame to store all the data
    df_all = pd.DataFrame()

    # Iterate through the date range month by month
    current_start = start_date
    i = 0
    params = {
        "adjusted": "true",   # Get adjusted stock prices
        "sort": "asc",        # Sort the data in ascending order
        "limit": 50000,       # Fetch as much data as possible
        "apiKey": api_key     # Polygon.io API key
    }
    while current_start < end_date:
        # Define the start and end of the current month
        current_end = current_start + timedelta(days=90)
        if current_end > end_date:
            current_end = end_date

        # Convert dates to string format for the API
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')

        # API request URL for 2-minute interval stock prices
        url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_str}/{end_str}'
        print(url)

        while url:
            # Make the API request
            try:
                response = requests.get(url, params=params)
            except Exception as ex:
                print(ex)
                time.sleep(10)
                continue

            data = response.json()

            # Extract the results (each result corresponds to a 2-minute bar)
            results = data.get('results', [])
            status = data.get('status', None)
            err = data.get('error', None)
            if len(results) == 0 and status is not None and err is not None:
                print(data)
                t.sleep(70)
                continue

            # If there are results, convert to DataFrame and append to the main DataFrame
            if results:
                df = pd.DataFrame(results)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                if symbol.startswith('O:'):
                    _, ticker = symbol.split(':')
                    sym, exp, typ, strike = parseOptionTicker(symbol.split(':')[1])
                    df['strike'] = strike
                    df['sym'] = ticker

                    df = df[['timestamp', 'o', 'h', 'l', 'c', 'v', 'vw', 'n', 'strike', 'sym']]
                    df.columns = ['Datetime', 'Open', 'High', 'Low',
                                  'Close', 'Volume', 'VolumeWeighted', 'Trasaction', 'Strike', 'Symbol']
                    df_all = pd.concat([df_all, df], ignore_index=True)
                else:
                    df = df[['timestamp', 'o', 'h', 'l', 'c', 'v', 'vw', 'n']]
                    df.columns = ['Datetime', 'Open', 'High', 'Low',
                                  'Close', 'Volume', 'VolumeWeighted', 'Trasaction']
                    df_all = pd.concat([df_all, df], ignore_index=True)

            # Check if there is a 'next_url' for pagination, otherwise exit the loop
            url = data.get('next_url', None)

        # Move to the next month
        current_start = current_end
        i += 1
        if i % 5 == 0:
            print("sleep")
            t.sleep(70)

    if not df_all.empty:
        # Set the 'Datetime' column as the DataFrame index
        df_all.set_index('Datetime', inplace=True)

        # Optionally, sort the index
        df_all.sort_index(inplace=True)

        # Localize index to UTC
        df_all.index = df_all.index.tz_localize('UTC')

        # Convert the index to Eastern Time
        df_all.index = df_all.index.tz_convert('US/Eastern')

        # Define the stock market trading hours (09:30 to 16:00 Eastern Time)
        start_time = time(9, 30)
        end_time = time(16, 0)
        return df_all[(df_all.index.time >= start_time) & (df_all.index.time <= end_time)]
    return df_all
