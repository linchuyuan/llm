#!/usr/bin/env python3

from datetime import datetime, timedelta
import requests
import pdb
import pandas as pd
from datetime import time
import time as t
import os

# api_key = 'gvhZSVGOB2_lwiCC71sngkBa8OtXKctd'
# Stock symbol (AAPL in this case)
# symbol = 'AAPL'
# Calculate the date range: from one year ago to today
# end_date = datetime.now()
# start_date = end_date - timedelta(days=360)
def getStockHistory(symbol, api_key=None, start_date=None, duration=None):
    if duration is None:
        duration = timedelta(days=360)
    if api_key is None:
        api_key = os.environ['api_key']

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
            response = requests.get(url, params=params)
            data = response.json()

            # Extract the results (each result corresponds to a 2-minute bar)
            results = data.get('results', [])
            if len(results) == 0:
                print(data)
                t.sleep(70)
                continue

            # If there are results, convert to DataFrame and append to the main DataFrame
            if results:
                df = pd.DataFrame(results)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df = df[['timestamp', 'o', 'h', 'l', 'c', 'v', 'vw', 'n']]
                df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeWeighted', 'Trasaction']
                df_all = pd.concat([df_all, df], ignore_index=True)

            # Check if there is a 'next_url' for pagination, otherwise exit the loop
            url = data.get('next_url', None)

        # Move to the next month
        current_start = current_end
        i += 1
        if i % 5 == 0:
            print("sleep")
            t.sleep(70)

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

    # Filter the DataFrame index (primary index) to include only trading hours
    return df_all[(df_all.index.time >= start_time) & (df_all.index.time <= end_time)]
