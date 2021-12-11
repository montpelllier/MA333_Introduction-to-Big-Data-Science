import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats

path1 = 'project_data/DC_Crime.csv'
path2 = 'project_data/DC_Properties.csv'
path3 = 'project_data/DC_crime_test.csv'

df = pd.read_csv(path1, low_memory=False)
df2 = pd.read_csv(path2, low_memory=False)

drop_list = ['CMPLX_NUM','LIVING_GBA','STYLE','USECODE','GIS_LAST_MOD_DTTM','SOURCE',\
         'FULLADDRESS','CITY','STATE','NATIONALGRID','LATITUDE','LONGITUDE','ASSESSMENT_SUBNBHD',\
         'QUADRANT']
df2.drop(drop_list,axis=1,inplace=True)


def getYear(time_str):
    return int(time_str[:4])

new_df = df2[df2['PRICE']>500]
new_df.dropna(axis=0, how='any',inplace=True) #去除空值
year = new_df[['SALEDATE']]
year['SALEDATE'] = new_df['SALEDATE'].apply(getYear)
new_df['SALEDATE'] = year['SALEDATE']

year_price = new_df.groupby(['SALEDATE'])['PRICE'].mean()
d = dict(year_price)
plt.plot(list(d.keys()), list(d.values()))
plt.show()

year_price = pd.DataFrame()
year_price['YEAR'] = d.keys()
year_price['Price'] = d.values()
df = pd.merge(df,year_price)

r,p = stats.pearsonr(df.Price,df["ucr-rank"])
print('相关系数r为 = %6.3f，p值为 = %6.3f'%(r,p))

cnt = df.groupby(['YEAR'])['YEAR'].count()
d = dict(cnt)
cnt = pd.DataFrame()
cnt['YEAR'] = d.keys()
cnt['cnt'] = d.values()
x = pd.merge(cnt,year_price)
r,p = stats.pearsonr(x.Price,x.cnt)
print('相关系数r为 = %6.3f，p值为 = %6.3f'%(r,p))
