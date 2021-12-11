import pandas as pd
import datetime

start_time = datetime.datetime.now()  # 开始计时

path1 = 'project_data/DC_Crime.csv'
path2 = 'project_data/DC_Properties.csv'
path3 = 'project_data/DC_crime_test.csv'

df = pd.read_csv(path1, low_memory=False)
print(df.info())
df = pd.read_csv(path2, low_memory=False)
print(df.info())
print(datetime.datetime.now() - start_time)