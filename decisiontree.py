import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

path1 = 'project_data/DC_Crime.csv'
path2 = 'project_data/DC_Properties.csv'
path3 = 'project_data/DC_crime_test.csv'
data = pd.read_csv(path1)

Features =['SHIFT', 'OFFENSE', 'METHOD','BID',"NEIGHBORHOOD_CLUSTER",'ucr-rank',\
           'sector','ANC','BLOCK_GROUP','BLOCK', 'DISTRICT','location','offensegroup',\
           'PSA','WARD','VOTING_PRECINCT','CCN','END_DATE','OCTO_RECORD_ID','offense-text',\
           'offensekey', 'XBLOCK', 'YBLOCK', 'START_DATE','REPORT_DAT','CENSUS_TRACT']
X = data.drop(columns=Features)
y= data['offensegroup']
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify =y)

imp= SimpleImputer()
X_train = imp.fit_transform(X_train)

X_test = imp.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=50, random_state=40)
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
treefeatures=dtree.feature_importances_
print(dtree.score(X_train,y_train))
print(list(y_pred))
#预测用dtree.predict,data是导入数据
