import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('global_temps.csv')

season_mapping = {
    'DJF': 'Winter',
    'MAM': 'Spring',
    'JJA': 'Summer',
    'SON': 'Fall'
}

# Set the seaborn color palette for seasons
palette = sns.color_palette("husl", len(season_mapping))

# Prepare the data for plotting
seasonal_data = data[['Year', 'DJF', 'MAM', 'JJA', 'SON']].melt(id_vars='Year', 
                                                                 value_vars=['DJF', 'MAM', 'JJA', 'SON'],
                                                                 var_name='Season', 
                                                                 value_name='Temperature')

# Replace season codes with names
seasonal_data['Season'] = seasonal_data['Season'].map(season_mapping)

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
sns.lineplot(data=seasonal_data, x='Year', y='Temperature',marker='o', hue='Season', palette=palette)
plt.title('Global Temperatures by Season Over Time')
plt.xlabel('Year', color='white')
plt.ylabel('Temperature Anomaly (°C)', color='white')
plt.legend(title='Season')
plt.grid()
plt.legend()
plt.show()

# RANDOM FOREST REGRESSION

print("Checking for NaN values in the dataset:")
print(data.isnull().sum())

data = data.dropna(subset=['DJF'])

x = data[['Year']]  # Feature
y = data['DJF']     # Target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
RandomForest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
RandomForest_regressor.fit(x_train, y_train)

#predictions 
y_prediction = RandomForest_regressor.predict(x_test)
print(y_prediction)

# Evaluate the model
Mean_Square_Error = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)
print(f'Mean Squared Error: {Mean_Square_Error}')
print(f'R^2 Score: {r2}')

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
plt.scatter(x_test, y_test, color='cyan', label='Actual Values')  
plt.scatter(x_test, y_prediction, color='yellow', label='Predicted Values')  
plt.title('Actual vs Predicted Values')
plt.xlabel('Year', color='white')
plt.ylabel('DJF', color='white')
plt.legend()
plt.grid(color='gray')  
plt.show()

# RANDOM FOREST CLASSIFIER

print("Checking for NaN values in the dataset:")
print(data.isnull().sum())

data = data.dropna(subset=['DJF'])

# Create a binary target variable (1 if DJF > 0, else 0)
data['DJF_binary'] = (data['DJF'] > 0).astype(int)

x = data[['Year']]  # Feature
y = data['DJF_binary']  # Binary target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)

# Making predictions
y_prediction = rf_classifier.predict(x_test)
print("Predictions:")
print(y_prediction)

# Evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_prediction))
print("Classification Report:")
print(classification_report(y_test, y_prediction))

# Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
plt.scatter(x_test, y_test, color='yellow', label='Actual', alpha=0.6)
plt.scatter(x_test, y_prediction, color='green', label='Predicted', marker='x')
plt.xlabel('Year', color='white')
plt.ylabel('Temperature Anomaly (Binary)', color='white')
plt.title('Actual vs Predicted Temperature Anomaly')
plt.legend()
plt.grid()
plt.show()

# FINDING OUTLIERS

Q1 = data['DJF'].quantile(0.25)
Q3 = data['DJF'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['DJF'] < lower_bound) | (data['DJF'] > upper_bound)]

print("Outliers based on Djf values:")
print(outliers)

plt.figure(figsize=(12, 6))
plt.style.use('dark_background')  
plt.plot(data.index, data['DJF'], label='DJF Values', color='blue')
plt.scatter(outliers.index, outliers['DJF'], color='red', label='Outliers', marker='o')

plt.title('D-N Values with Outliers Highlighted')
plt.xlabel('Year', color='white')
plt.ylabel('DJF Values', color='white')
plt.axhline(y=lower_bound, color='green', linestyle='--', label='Lower Bound')
plt.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
plt.legend()
plt.grid(True)
plt.show()

# ARIMA

# Convert 'Year' to datetime format & set it as the Index to prepare the data
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

TimeSeries = data['DJF']
print("ARIMA")
print(TimeSeries)

TimeSeries = data['DJF'].dropna()  
adf_result = adfuller(TimeSeries)

print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value}')
    

TimeSeries = data['DJF'].dropna()  
def plot_acf_pacf(time_series):
    """Function to plot ACF and PACF for a given time series."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot ACF
    plot_acf(time_series, ax=axes[0], lags=30)
    axes[0].set_title('Autocorrelation Function (ACF)')

    # Plot PACF
    plot_pacf(time_series, ax=axes[1], lags=30)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()

# Call the function to plot ACF and PACF
plot_acf_pacf(TimeSeries)

# Assuming 'DJF' is the time series you want to analyze
TimeSeries = data['DJF'].dropna()  
TimeSeries_diff = TimeSeries.diff().dropna()

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
plt.plot(TimeSeries, label='First Difference of DJF', color='Orange')
plt.title('First Difference of DJF Time Series')
plt.xlabel('Year', color='white')
plt.ylabel('Difference', color='white')
plt.legend()
plt.grid()
plt.show()

# FIT ARIMA MODEL

ARIMA_model = ARIMA(TimeSeries, order=(1, 1, 1))
ARIMA_model_fit = ARIMA_model.fit()

print("ARIMA Model")
print(ARIMA_model_fit.summary())

# RESIDUAL,DENSITY,TREND

data = data.dropna(subset=['DJF'])
TimeSeries = data['DJF']

ARIMA_model = ARIMA(TimeSeries, order=(1, 1, 1))  # (p, d, q) order for ARIMA
ARIMA_result = ARIMA_model.fit()

# Forecast using the ARIMA model
forecast = ARIMA_result.predict(start=TimeSeries .index[0], end=TimeSeries .index[-1], typ='levels')

# Compute the residuals
residuals = TimeSeries  - forecast
print(residuals)

# Plotting the residuals
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
plt.plot(TimeSeries.index, residuals, color='purple', alpha=0.6)
plt.axhline(0, color='white', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residual Plot of ARIMA Model')
plt.show()

# Plotting the density of residuals
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
sns.kdeplot(residuals, shade=True, color="blue")
plt.xlabel('Residuals', color='white')
plt.title('Density Plot of Residuals')
plt.show()

# Plotting the trend: actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  
plt.plot(TimeSeries.index, TimeSeries, color='blue', label='Actual')
plt.plot(TimeSeries.index, forecast, color='red', linestyle='--', label='Forecasted')
plt.xlabel('Year', color='white')
plt.ylabel('Temperature Anomaly (DJF)')
plt.title('Trend Plot: Actual vs Forecasted (ARIMA)')
plt.legend()
plt.show()
