import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib
from sklearn.metrics import mean_squared_error
from pylab import rcParams
from fbprophet import Prophet

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df_confirmed_cases = pd.read_csv("/home/sharif/Documents/Intelligent Computing/All datasets/novel-corona-virus-2019-dataset/time_series-ncov-Confirmed.csv")
df_death_cases = pd.read_csv("/home/sharif/Documents/Intelligent Computing/All datasets/novel-corona-virus-2019-dataset/time_series-ncov-Deaths.csv")
df_recovery_cases = pd.read_csv("/home/sharif/Documents/Intelligent Computing/All datasets/novel-corona-virus-2019-dataset/time_series-ncov-Recovered.csv")
df_confirmed_cases=df_confirmed_cases.dropna(how='all')
df_death_cases=df_death_cases.dropna(how='all')
df_recovery_cases=df_recovery_cases.dropna(how='all')

#Time series analysis by facebook prophet
df_confirmed_cases = df_confirmed_cases.rename(columns={'Date': 'ds', 'Value': 'y'})
df_confirmed_cases_model = Prophet(interval_width=0.95)
df_confirmed_cases_model.fit(df_confirmed_cases)
df_death_cases =df_death_cases.rename(columns={'Date': 'ds', 'Value': 'y'})
df_death_cases_model = Prophet(interval_width=0.95)
df_death_cases_model.fit(df_death_cases)
df_recovery_cases =df_recovery_cases.rename(columns={'Date': 'ds', 'Value': 'y'})
df_recovery_cases_model = Prophet(interval_width=0.95)
df_recovery_cases_model.fit(df_recovery_cases)

df_confirmed_cases_forecast = df_confirmed_cases_model.make_future_dataframe(periods=36, freq='D')
df_confirmed_cases_forecast = df_confirmed_cases_model.predict(df_confirmed_cases_forecast)
df_death_cases_forecast = df_death_cases_model.make_future_dataframe(periods=36, freq='D')
df_death_cases_forecast = df_death_cases_model.predict(df_death_cases_forecast)
df_recovery_cases_forecast = df_recovery_cases_model.make_future_dataframe(periods=36, freq='D')
df_recovery_cases_forecast = df_recovery_cases_model.predict(df_recovery_cases_forecast)

#Compare Forecasts
df_confirmed_cases_names = ['df_confirmed_cases_%s' % column for column in df_confirmed_cases_forecast.columns]
df_death_cases_names = ['df_death_cases_%s' % column for column in df_death_cases_forecast.columns]
df_recovery_cases_names = ['df_recovery_cases_%s' % column for column in df_recovery_cases_forecast.columns]

merge_df_confirmed_cases_forecast = df_confirmed_cases_forecast.copy()
merge_df_death_cases_forecast = df_death_cases_forecast.copy()
merge_df_recovery_cases_forecast = df_recovery_cases_forecast.copy()

merge_df_confirmed_cases_forecast.columns = df_confirmed_cases_names
merge_df_death_cases_forecast.columns = df_death_cases_names
merge_df_recovery_cases_forecast.columns = df_recovery_cases_names

forecast_confirmed_death = pd.merge(merge_df_confirmed_cases_forecast, merge_df_death_cases_forecast, how = 'inner', left_on = 'df_confirmed_cases_ds', right_on = 'df_death_cases_ds')
forecast_confirmed_recovery = pd.merge(merge_df_confirmed_cases_forecast, merge_df_recovery_cases_forecast, how = 'inner', left_on = 'df_confirmed_cases_ds', right_on = 'df_recovery_cases_ds')
forecast_death_recovery = pd.merge(merge_df_death_cases_forecast, merge_df_recovery_cases_forecast, how = 'inner', left_on = 'df_death_cases_ds', right_on = 'df_recovery_cases_ds')

print(forecast_confirmed_death.head())
print(forecast_confirmed_recovery.head())
print(forecast_death_recovery.head())

#Estimation and Forecast Visualization for Confirmed Cases vs. Death Cases Estimate
plt.figure(figsize=(10, 7)) 
plt.plot(forecast_confirmed_death['df_confirmed_cases_ds'], forecast_confirmed_death['df_confirmed_cases_yhat'], 'b-')
plt.plot(forecast_confirmed_death['df_confirmed_cases_ds'], forecast_confirmed_death['df_death_cases_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Cases')
plt.title('Confirmed Cases vs. Death Cases Estimate');

#Estimation and Forecast Visualization for Confirmed Cases vs. Recovery Cases Estimate
plt.figure(figsize=(10, 7)) 
plt.plot(forecast_confirmed_recovery['df_confirmed_cases_ds'], forecast_confirmed_recovery['df_confirmed_cases_yhat'], 'b-')
plt.plot(forecast_confirmed_recovery['df_confirmed_cases_ds'], forecast_confirmed_recovery['df_recovery_cases_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Cases')
plt.title('Confirmed Cases vs. Recovery Cases Estimate');

#Estimation and Forecast Visualization for Death Cases vs. Recovery Cases Estimate
plt.figure(figsize=(10, 7)) 
plt.plot(forecast_death_recovery['df_death_cases_ds'], forecast_death_recovery['df_death_cases_yhat'], 'b-')
plt.plot(forecast_death_recovery['df_death_cases_ds'], forecast_death_recovery['df_recovery_cases_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Cases')
plt.title('Death Cases vs. Recovery Cases Estimate');







