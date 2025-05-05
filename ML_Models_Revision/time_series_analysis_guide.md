# Comprehensive Time Series Analysis and Forecasting Guide

This guide provides detailed explanations and implementations for time series analysis, from fundamentals to advanced techniques.

## Table of Contents
1. [Introduction to Time Series](#introduction-to-time-series)
2. [Key Characteristics](#key-characteristics)
3. [Common Use Cases](#common-use-cases)
4. [Data Preparation](#data-preparation)
5. [Exploratory Time Series Analysis](#exploratory-time-series-analysis)
6. [Statistical Models](#statistical-models)
7. [Machine Learning Models](#machine-learning-models)
8. [Deep Learning Models](#deep-learning-models)
9. [Model Evaluation](#model-evaluation)
10. [Handling Multiple Time Series](#handling-multiple-time-series)
11. [Real-Life Challenges and Solutions](#real-life-challenges-and-solutions)
12. [Interview Questions and Answers](#interview-questions-and-answers)

## Introduction to Time Series

Time series analysis is the study of data points collected or recorded at specific time intervals. Unlike regular data analysis, time series data has a natural temporal ordering, making the time dimension a crucial component of the analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load example time series data
# Using a common dataset: airline passengers
from statsmodels.datasets import get_rdataset
airline = get_rdataset('AirPassengers').data
airline.index = pd.date_range(start='1949-01-01', periods=len(airline), freq='MS')
airline.columns = ['passengers']

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(airline.index, airline['passengers'])
plt.title('Monthly Airline Passenger Numbers 1949-1960')
plt.xlabel('Date')
plt.ylabel('Passengers (thousands)')
plt.grid(True)
plt.show()
```

## Key Characteristics

### 1. Temporal Dependency

```python
# Visualize autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))
plot_acf(airline['passengers'], lags=36)
plt.title('Autocorrelation Function')
plt.show()

# Calculate correlation with lagged values
for lag in [1, 6, 12]:
    correlation = airline['passengers'].corr(airline['passengers'].shift(lag))
    print(f"Correlation with lag {lag}: {correlation:.4f}")
```

### 2. Seasonality

```python
# Decompose time series into trend, seasonal, and residual components
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(airline['passengers'], model='multiplicative', period=12)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonality')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residuals')
plt.tight_layout()
plt.show()
```

### 3. Trend

```python
# Calculate rolling average to visualize trend
rolling_mean = airline['passengers'].rolling(window=12).mean()

plt.figure(figsize=(12, 6))
plt.plot(airline.index, airline['passengers'], label='Original')
plt.plot(airline.index, rolling_mean, label='12-month rolling average', color='red')
plt.title('Airline Passengers with Trend')
plt.legend()
plt.grid(True)
plt.show()
```

### 4. Cyclical Patterns and Irregularity

```python
# Create a more complex dataset with cycle and noise
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', periods=1000, freq='D')
trend = np.linspace(0, 30, 1000)
seasonality = 5 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
cycle = 7 * np.sin(2 * np.pi * np.arange(1000) / 1000)
noise = np.random.normal(0, 2, 1000)

ts = pd.Series(trend + seasonality + cycle + noise, index=dates)

plt.figure(figsize=(12, 6))
plt.plot(ts)
plt.title('Time Series with Trend, Seasonality, Cycle, and Noise')
plt.grid(True)
plt.show()
```

## Common Use Cases

### 1. Financial Forecasting

```python
# Example: Stock price forecasting
import yfinance as yf

# Download stock data
stock_data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')

# Plot closing prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'])
plt.title('Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()

# Calculate daily returns
stock_data['Returns'] = stock_data['Close'].pct_change() * 100

# Plot returns
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Returns'])
plt.title('Daily Returns (%)')
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.grid(True)
plt.show()
```

### 2. Demand Planning

```python
# Example: Sales forecasting
# Create synthetic sales data with trend and seasonality
dates = pd.date_range(start='2018-01-01', periods=365*3, freq='D')
trend = np.linspace(100, 200, 365*3)
seasonality = 30 * np.sin(2 * np.pi * np.arange(365*3) / 365.25)
weekly = 15 * np.sin(2 * np.pi * np.arange(365*3) / 7)
noise = np.random.normal(0, 10, 365*3)

sales = pd.Series(trend + seasonality + weekly + noise, index=dates)
sales = sales.clip(lower=0)  # No negative sales

# Plot sales data
plt.figure(figsize=(12, 6))
plt.plot(sales)
plt.title('Daily Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales Units')
plt.grid(True)
plt.show()

# Aggregate to monthly for easier visualization
monthly_sales = sales.resample('M').sum()
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales)
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales Units')
plt.grid(True)
plt.show()
```

### 3. Energy Management

```python
# Example: Electricity load forecasting
# Create synthetic hourly electricity load data
hours = pd.date_range(start='2022-01-01', periods=24*365, freq='H')
base_load = 1000
daily_pattern = 300 * np.sin(2 * np.pi * np.arange(24*365) / 24 - np.pi/2)
weekly_pattern = 100 * np.sin(2 * np.pi * np.arange(24*365) / (24*7))
seasonal_pattern = 200 * np.sin(2 * np.pi * np.arange(24*365) / (24*365))
noise = np.random.normal(0, 50, 24*365)

load = pd.Series(base_load + daily_pattern + weekly_pattern + seasonal_pattern + noise, index=hours)
load = load.clip(lower=0)

# Plot one week of hourly data
plt.figure(figsize=(12, 6))
plt.plot(load.iloc[:24*7])
plt.title('Hourly Electricity Load (One Week)')
plt.xlabel('Date')
plt.ylabel('Load (MW)')
plt.grid(True)
plt.show()

# Plot daily aggregated data for the year
daily_load = load.resample('D').mean()
plt.figure(figsize=(12, 6))
plt.plot(daily_load)
plt.title('Daily Average Electricity Load')
plt.xlabel('Date')
plt.ylabel('Load (MW)')
plt.grid(True)
plt.show()
```

## Data Preparation

### 1. Handling Missing Values

```python
# Create a time series with missing values
ts = pd.Series(np.random.normal(0, 1, 100), index=pd.date_range('2023-01-01', periods=100))
# Introduce missing values
ts[10:15] = np.nan
ts[60:62] = np.nan
ts[85] = np.nan

print("Time series with missing values:")
print(ts.isna().sum(), "missing values")

# Method 1: Forward fill
ts_ffill = ts.ffill()

# Method 2: Backward fill
ts_bfill = ts.bfill()

# Method 3: Linear interpolation
ts_interp = ts.interpolate(method='linear')

# Method 4: Spline interpolation
ts_spline = ts.interpolate(method='spline', order=3)

# Method 5: Time-weighted interpolation
ts_time = ts.interpolate(method='time')

# Visualize the different methods
plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(ts, 'o-', label='Original with NaNs')
plt.title('Original Data with Missing Values')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(ts_ffill, 'o-', label='Forward Fill')
plt.title('Forward Fill')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(ts_bfill, 'o-', label='Backward Fill')
plt.title('Backward Fill')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(ts_interp, 'o-', label='Linear Interpolation')
plt.title('Linear Interpolation')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(ts_spline, 'o-', label='Spline Interpolation')
plt.title('Spline Interpolation')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(ts_time, 'o-', label='Time Interpolation')
plt.title('Time Interpolation')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. Ensuring Consistent Frequency

```python
# Create a time series with irregular timestamps
irregular_dates = pd.date_range('2023-01-01', periods=50, freq='D').tolist()
# Add some irregular gaps
irregular_dates = irregular_dates[:20] + pd.date_range('2023-02-01', periods=15, freq='2D').tolist() + irregular_dates[35:]
irregular_values = np.random.normal(10, 2, len(irregular_dates))

irregular_ts = pd.Series(irregular_values, index=irregular_dates)
print("Original irregular time series:")
print(irregular_ts.head())

# Resample to daily frequency
daily_ts = irregular_ts.resample('D').mean()

# Fill missing values after resampling
daily_ts_filled = daily_ts.interpolate(method='linear')

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(irregular_ts.index, irregular_ts.values, 'o-', label='Irregular')
plt.plot(daily_ts_filled.index, daily_ts_filled.values, '-', label='Regular (Daily)')
plt.title('Resampling to Regular Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Upsampling example (daily to hourly)
hourly_ts = irregular_ts.resample('H').interpolate(method='cubic')

# Downsampling example (to weekly)
weekly_ts = irregular_ts.resample('W').mean()

print("After resampling to weekly:")
print(weekly_ts.head())
```

### 3. Outlier Detection and Handling

```python
# Create a time series with outliers
np.random.seed(42)
ts = pd.Series(np.random.normal(10, 2, 100), index=pd.date_range('2023-01-01', periods=100))
# Add outliers
ts[25] = 30
ts[60] = -10
ts[80] = 25

# Method 1: Z-score method
from scipy import stats

z_scores = stats.zscore(ts)
abs_z_scores = np.abs(z_scores)
outlier_indices = np.where(abs_z_scores > 3)[0]
print(f"Outliers detected at indices: {outlier_indices}")

# Method 2: IQR method
Q1 = ts.quantile(0.25)
Q3 = ts.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
iqr_outliers = ts[(ts < lower_bound) | (ts > upper_bound)]
print(f"IQR outliers detected at indices: {iqr_outliers.index.tolist()}")

# Method 3: Rolling median and standard deviation
rolling_median = ts.rolling(window=10, center=True).median()
rolling_std = ts.rolling(window=10, center=True).std()
outliers = ts[(ts > rolling_median + 3*rolling_std) | (ts < rolling_median - 3*rolling_std)]
print(f"Rolling window outliers: {outliers.index.tolist()}")

# Visualize outliers
plt.figure(figsize=(12, 6))
plt.plot(ts, 'o-', label='Original')
plt.plot(ts.index[outlier_indices], ts.iloc[outlier_indices], 'ro', markersize=10, label='Z-score Outliers')
plt.plot(iqr_outliers.index, iqr_outliers, 'go', markersize=8, label='IQR Outliers')
plt.plot(outliers.index, outliers, 'mo', markersize=6, label='Rolling Window Outliers')
plt.title('Time Series with Outliers')
plt.legend()
plt.grid(True)
plt.show()

# Handle outliers by replacing with rolling median
ts_cleaned = ts.copy()
ts_cleaned[outlier_indices] = rolling_median.iloc[outlier_indices]

plt.figure(figsize=(12, 6))
plt.plot(ts, 'o-', alpha=0.5, label='Original')
plt.plot(ts_cleaned, 'o-', label='Cleaned')
plt.title('Time Series with Outliers Removed')
plt.legend()
plt.grid(True)
plt.show()
```

### 4. Stationarity

```python
# Check for stationarity
from statsmodels.tsa.stattools import adfuller

def check_stationarity(ts, window=12):
    # Rolling statistics
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original')
    plt.plot(rolling_mean, label=f'Rolling Mean ({window})')
    plt.plot(rolling_std, label=f'Rolling Std ({window})')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid(True)
    plt.show()
    
    # Augmented Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    if dftest[1] <= 0.05:
        print("Conclusion: Time series is stationary")
    else:
        print("Conclusion: Time series is non-stationary")

# Check airline data for stationarity
check_stationarity(airline['passengers'])

# Make the series stationary
# Method 1: Differencing
airline_diff = airline['passengers'].diff().dropna()
check_stationarity(airline_diff)

# Method 2: Log transformation + differencing
airline_log = np.log(airline['passengers'])
airline_log_diff = airline_log.diff().dropna()
check_stationarity(airline_log_diff)

# Method 3: Seasonal differencing
airline_seasonal_diff = airline['passengers'].diff(12).dropna()
check_stationarity(airline_seasonal_diff)
```

### 5. Feature Engineering

```python
# Create time-based features
def create_time_features(df, target_col):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
    # Cyclical encoding for hour, day, month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Lag features
    for lag in [1, 7, 28]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling window features
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
       