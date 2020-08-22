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

print("Confirmed Cases Dataset Head : ")
print(df_confirmed_cases.head())
print("Death Cases Dataset Head : ")
print(df_death_cases.head())
print("Recovery Cases Dataset Head : ")
print(df_recovery_cases.head())

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

plt.figure(figsize=(18, 6))
df_confirmed_cases_model.plot(df_confirmed_cases_forecast, xlabel = 'Date', ylabel = 'Confirm Cases')
plt.title('COVID 19 Confirmed Cases');
plt.figure(figsize=(18, 6))
df_death_cases_model.plot(df_death_cases_forecast, xlabel = 'Date', ylabel = 'Death Cases')
plt.title('COVID 19 Death Cases');
plt.figure(figsize=(18, 6))
df_recovery_cases_model.plot(df_recovery_cases_forecast, xlabel = 'Date', ylabel = 'Recovery Cases')
plt.title('COVID 19 Recovery Cases');








