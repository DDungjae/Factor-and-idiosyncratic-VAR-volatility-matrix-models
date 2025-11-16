import pandas as pd
import numpy as np
import os
from datetime import datetime

# Create log_return directory if it doesn't exist
if not os.path.exists('log_return'):
    os.makedirs('log_return')

# Read the data
print("Reading the data file...")
encoding = "latin1"  # Changed to latin1 encoding which can read any byte
df = pd.read_csv('2015_2020_1min.csv', encoding=encoding)
print(f"Successfully read the file with {encoding} encoding")

# Data preprocessing
print("\nPreprocessing data...")

# Print first few rows to check data format
print("\nFirst few rows of the data:")
print(df.head())

# Convert timestamp column to datetime
timestamp_col = df.columns[0]
print(f"\nTimestamp column name: {timestamp_col}")

# Print unique values in timestamp column to check for anomalies
print("\nUnique values in timestamp column:")
print(df[timestamp_col].unique()[:10])  # Print first 10 unique values

# Try to convert timestamp with error handling
try:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    # Check for any NaT (Not a Time) values
    nat_count = df[timestamp_col].isna().sum()
    if nat_count > 0:
        print(f"\nWarning: Found {nat_count} invalid timestamps")
        # Print rows with invalid timestamps
        invalid_rows = df[df[timestamp_col].isna()]
        print("\nFirst few rows with invalid timestamps:")
        print(invalid_rows.head())
except Exception as e:
    print(f"\nError converting timestamps: {str(e)}")
    raise

# Get price columns (excluding timestamp)
price_cols = df.columns[1:]

# Convert price columns to numeric
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate number of chunks (390 rows per chunk)
n_chunks = int(len(df) // 390)  # Ensure integer division

print(f"\nSplitting data into {n_chunks} daily chunks...")

# Process each daily chunk
for i in range(n_chunks):
    # Calculate start and end indices
    start_idx = int(i * 390)  # Ensure integer
    end_idx = int((i + 1) * 390)  # Ensure integer
    
    # Extract daily chunk
    daily_data = df.iloc[start_idx:end_idx].copy()
    
    # Calculate log returns
    log_returns = pd.DataFrame()
    log_returns[timestamp_col] = daily_data[timestamp_col]
    
    for col in price_cols:
        # Forward fill NaN values in prices
        prices = daily_data[col].fillna(method='ffill')
        
        # If there are still NaN values at the start, fill with 0
        prices = prices.fillna(0)
        
        # Calculate log returns
        log_returns[col] = np.log(prices / prices.shift(1))
        
        # Replace infinite values with NaN
        log_returns[col] = log_returns[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with 0
        log_returns[col] = log_returns[col].fillna(0)
    
    # Remove first row (NaN due to log returns calculation)
    log_returns = log_returns.iloc[1:]
    
    # Save to CSV
    output_file = f'log_return/d{i+1}.csv'
    log_returns.to_csv(output_file, index=False, encoding='utf-8')
    
    # Print progress
    print(f"Created {output_file} ({len(log_returns)} rows)")

print(f"\nTotal daily chunks created: {n_chunks}")
print("Done!")
