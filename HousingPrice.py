import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, SGDRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
import datetime

start_time = datetime.datetime.now()  # 开始计时

path1 = 'project_data/DC_Crime.csv'
path2 = 'project_data/DC_Properties.csv'
path3 = 'project_data/DC_crime_test.csv'


def getYear(time_str):
    return int(time_str[:4])


df = pd.read_csv(path2, low_memory=False)
data = pd.read_csv(path1)
drop_list = ['CMPLX_NUM','LIVING_GBA','STYLE','USECODE','GIS_LAST_MOD_DTTM','SOURCE',\
         'FULLADDRESS','CITY','STATE','NATIONALGRID','LATITUDE','LONGITUDE','ASSESSMENT_SUBNBHD',\
         'QUADRANT']
df.drop(drop_list,axis=1,inplace=True)
Features =['SHIFT', 'OFFENSE', 'METHOD','BID',"NEIGHBORHOOD_CLUSTER",'ucr-rank',\
           'sector','ANC','BLOCK_GROUP','BLOCK', 'DISTRICT','location','offensegroup',\
           'PSA','WARD','VOTING_PRECINCT','CCN','END_DATE','OCTO_RECORD_ID','offense-text',\
           'offensekey', 'XBLOCK', 'YBLOCK', 'START_DATE','REPORT_DAT','CENSUS_TRACT']
X = data.drop(columns=Features)

print(df['PRICE'].describe())
new_df = df[df['PRICE']>500]
new_df.dropna(axis=0, how='any',inplace=True) #去除空值

year = new_df[['SALEDATE']]
year['SALEDATE'] = year['SALEDATE'].apply(getYear)
new_df['SALEDATE'] = year['SALEDATE']

y= data['offensegroup']
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify =y)
imp= SimpleImputer()
X_train = imp.fit_transform(X_train)
X_test = new_df[['X','SALEDATE','Y']]
X_test = imp.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=50, random_state=40)
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
new_df['offense_group'] = list(y_pred)

price = new_df['PRICE']
print(price.describe())
sns.displot(price)
ax = plt.gca()
ax.set_xlim(0, 3.0e+06)
plt.show()


result = OneHotEncoder(categories='auto').fit_transform(new_df[['HEAT', 'STRUCT', 'EXTWALL', 'ROOF', 'INTWALL','WARD']]).toarray()
result.shape
result = pd.concat([new_df['PRICE'],pd.DataFrame(result)],axis=1)
cor_matrix = result.corr()
cor_list = list(cor_matrix['PRICE'])
print(cor_list)#名义型变量处理

print(new_df['AC'].value_counts())
print(new_df['QUALIFIED'].value_counts())
print(new_df['GRADE'].value_counts())
print(new_df['CNDTN'].value_counts())
print(new_df['offense_group'].value_counts())

grade_label = ['Fair Quality', 'Average', 'Above Average', 'Good Quality', 'Very Good','Excellent', \
               'Superior', 'Exceptional-A', 'Exceptional-B', 'Exceptional-C', 'Exceptional-D','No Data']
con_label = ['Poor', 'Fair', 'Average', 'Good', 'Very Good', 'Excellent','Default']
new_df['AC'] = new_df['AC'].replace(['Y','N'],[1,-1])
new_df['QUALIFIED'] = new_df['QUALIFIED'].replace(['Q','U'],[1,0])
new_df['offense_group'] = new_df['offense_group'].replace(['property','violent'],[1,0])
temp1 = [1,2,3,4,5,6,7,8,9,10,11,3.421669]
temp2 = [1,2,3,4,5,6,3.8559]
new_df['GRADE'] = new_df['GRADE'].replace(grade_label,temp1)
new_df['CNDTN'] = new_df['CNDTN'].replace(con_label,temp2)#等级型变量处理

sns.jointplot(x='GBA',y='PRICE',data=new_df)
sns.jointplot(x='STORIES',y='PRICE',data=new_df)

plt.figure(figsize=(15, 12))
cor_matrix = new_df.corr()
sns.heatmap(cor_matrix,square=True,vmax=1,vmin=-1,center=0.0,cmap='coolwarm')
plt.show()
print(cor_matrix['PRICE'])

#转换数据类型为float，并得到其相关系数矩阵
new_df[['AC','QUALIFIED','GRADE', 'CNDTN','SQUARE']].apply(lambda x:x.astype(float))
cor_matrix = new_df[['PRICE','AC','QUALIFIED','GRADE', 'CNDTN', 'SQUARE']].corr()
print(cor_matrix['PRICE'])

target_data = new_df[['PRICE']]
feature_data = new_df[['BATHRM','HF_BATHRM','NUM_UNITS','ROOMS','BEDRM',\
'AYB','YR_RMDL','EYB','STORIES','SALEDATE','SALE_NUM','GBA','BLDG_NUM',\
'KITCHENS','FIREPLACES','LANDAREA','ZIPCODE','CENSUS_TRACT','X','Y',\
'offense_group','AC','QUALIFIED','GRADE','CNDTN','SQUARE']]
model = LinearRegression()
model = model.fit(feature_data, target_data)
r_sq = model.score(feature_data, target_data)
print('coefficient of determination(𝑅²) :', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)#线性回归

ridgeRegression = Ridge()
ridgeRegression.fit(feature_data,target_data)
print(ridgeRegression.score(feature_data,target_data))
print(ridgeRegression.coef_)

Lambdas=np.logspace(-5,2,200)
lasso_cv=LassoCV(alphas=Lambdas,normalize=True,cv=10,max_iter=10000)
lasso_cv.fit(feature_data,target_data)
lasso = Lasso(alpha=lasso_cv.alpha_,normalize=True,max_iter=10000)
lasso.fit(feature_data,target_data)
print(lasso.score(feature_data,target_data))
print(lasso.coef_)

feature_data = new_df[['BATHRM', 'HF_BATHRM', 'ROOMS', 'BEDRM','SALEDATE' ,'EYB','offense_group',\
                       'GBA', 'FIREPLACES','LANDAREA', 'X', 'GRADE', 'CNDTN', 'SQUARE']]
model = LinearRegression()
model = model.fit(feature_data, target_data)
r_sq = model.score(feature_data, target_data)
print('coefficient of determination(𝑅²) :', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)#线性回归

print(datetime.datetime.now() - start_time)