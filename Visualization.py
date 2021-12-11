import datetime

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt, axes


def getDate(time_str):  # 将string转换为date
    return datetime.datetime.strptime(time_str[:7], "%Y-%m")

def getHour(time_str):
    return int(time_str[11:13])

def getMonth(time_str):
    return int(time_str[5:7])

path1 = 'project_data/DC_Crime.csv'
path2 = 'project_data/DC_Properties.csv'
path3 = 'project_data/DC_crime_test.csv'

start_time = datetime.datetime.now()  # 开始计时

df = pd.read_csv(path1)

plt.figure(figsize=(8, 5))  # num of offense by year
sns.countplot('YEAR', data=df)
plt.figure(figsize=(8, 5))  # num of offense by shift
sns.countplot(x='SHIFT', hue='offensegroup', data=df[['SHIFT', 'offensegroup']])
plt.show()
plt.figure(figsize=(12, 8))
sns.countplot(x='OFFENSE', hue='offensegroup', data=df[['OFFENSE', 'offensegroup']])
plt.show()
plt.figure(figsize=(12, 8))
sns.countplot(x='OFFENSE', hue='METHOD', data=df[['OFFENSE', 'METHOD']])
plt.show()
plt.figure(figsize=(12, 8))
sns.countplot(x='OFFENSE', hue='ucr-rank', data=df[['OFFENSE', 'ucr-rank']])
plt.show()

plt.title("Total", fontsize=24)  # 犯罪类型饼状图
dictionary = dict(zip(*np.unique(df['OFFENSE'], return_counts=True)))
plt.pie(x=dictionary.values(), labels=dictionary.keys(), autopct='%1.2f%%', shadow=True)
plt.show()

tmp = df[['OFFENSE', 'END_DATE']]
tmp.dropna(axis=0, how='any',inplace=True)
tmp['END_DATE'] = tmp['END_DATE'].apply(getDate)

offense_list = ['theft/other', 'theft f/auto', 'assault w/dangerous weapon', 'motor vehicle theft', 'robbery', \
                'burglary', 'sex abuse', 'homicide', 'arson']
offense_list = offense_list[6:]
for offense in offense_list:
    d = dict(tmp[tmp['OFFENSE'] == offense].groupby(tmp['END_DATE'])['OFFENSE'].count())
    plt.plot(list(d.keys()), list(d.values()))
date_min = datetime.datetime.strptime("2008-01", "%Y-%m")
date_max = datetime.datetime.strptime("2021-04", "%Y-%m")

ax = plt.gca()
ax.set_xlim(date_min, date_max)
plt.legend(offense_list, loc='best')# 随月份各种类犯罪变化图
plt.show()

hour = df[['END_DATE', 'METHOD']]
hour.dropna(axis=0, how='any',inplace=True)
hour['END_DATE'] = hour['END_DATE'].apply(getHour)
hour.rename(columns={'END_DATE':'Hour'},inplace=True)
sns.boxplot(x='METHOD',y='Hour',data=hour,palette='winter_r')
plt.show()

plt.figure(figsize=(15, 10))
hour = df[['END_DATE', 'OFFENSE']]
hour.dropna(axis=0, how='any',inplace=True)
hour['END_DATE'] = hour['END_DATE'].apply(getHour)
hour.rename(columns={'END_DATE':'Hour'},inplace=True)
sns.boxplot(x='OFFENSE',y='Hour',data=hour,palette='winter_r')
plt.show()

month = df[['END_DATE', 'ucr-rank']]
month.dropna(axis=0, how='any',inplace=True)
month['END_DATE'] = month['END_DATE'].apply(getMonth)
month.rename(columns={'END_DATE':'month'},inplace=True)
sns.countplot(y='month', hue='ucr-rank', data=month)
plt.show()


print(datetime.datetime.now() - start_time)
