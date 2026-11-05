# Flight Fare Prediction Using Machine Learning and Feature Engineering

## Project Overview
This project analyses flight booking data and develops regression models to estimate ticket prices. It investigates how airline, route, stops, journey date, departure and arrival times, and flight duration relate to ticket price.

## Objectives
- Explore relationships between flight attributes and ticket prices.
- Transform date, time, duration, and categorical features.
- Detect and handle outliers.
- Select useful features and train regression models.
- Tune and compare model performance.

## Dataset
The notebook uses `Data_Train.xlsx` and predicts the `Price` column. Features include airline, source, destination, total stops, route, journey date, departure and arrival times, and duration.

## Project Workflow
1. Explore flight attributes and their relationship with ticket price.
2. Prepare date, time, duration, and categorical variables.
3. Investigate and treat outliers.
4. Encode categorical variables and select useful features.
5. Train regression models and tune selected algorithms.

## Machine Learning Models
- Linear Regression
- K-Nearest Neighbours Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Evaluation
The notebook evaluates models using R² score, MAE, MSE, and RMSE. The notebook reports an XGBoost test R² score of approximately 83.87% after tuning.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- OpenPyXL

## Project Notebook
Open the project notebook: `Pro(3).ipynb`.

## Complete Project Code
The complete source code from the notebook is included below. Expand the section to view it.

<details>
<summary>View complete project code</summary>

### Code Cell 1

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```


### Code Cell 2

```python
df = pd.read_excel('Data_Train.xlsx')
```


### Code Cell 3

```python
df.head()
```


### Code Cell 4

```python
df.info()
```


### Code Cell 5

```python
df.describe()
```


### Code Cell 6

```python
df.tail()
```


### Code Cell 7

```python
df.head()
```


### Code Cell 8

```python
df1 = ['Airline','Source','Destination','Total_Stops']
```


### Code Cell 9

```python
plotnumber = 1
for col in df1:
    plt.subplot(2,2,plotnumber)
    sns.histplot(data=df,x=col)
    plt.xticks(rotation=90)
    plotnumber = plotnumber+1
```


### Code Cell 10

```python
plt.figure(figsize=(35,25),facecolor='white')
plotnumber=1
for col in df1:
    plt.subplot(2,2,plotnumber)
    sns.barplot(data=df,x=col,y=df['Price'])
    plt.xlabel(col,fontsize=30)
    plt.ylabel('Price',fontsize=30)
    plt.xticks(fontsize=20,rotation=90)
    plt.yticks(fontsize=20)
    plotnumber= plotnumber+1
plt.tight_layout()
```


### Code Cell 11

```python
sns.relplot(x='Total_Stops',y='Price',data=df)
plt.plot()
```


### Code Cell 12

```python
sns.relplot(x='Source',y='Price',col='Total_Stops',data=df,col_wrap=3)
plt.plot()
plt.xticks(rotation=90)
plt.tight_layout()
```


### Code Cell 13

```python
sns.histplot(df.Price,kde=True,bins=50)
plt.plot()
```


### Code Cell 14

```python
df.isnull().sum()
```


### Code Cell 15

```python
df.dropna(inplace = True)
df.isnull().sum()
```


### Code Cell 16

```python
df.info()
```


### Code Cell 17

```python
def change_into_datetime(col):
    df[col] = pd.to_datetime(df[col])
```


### Code Cell 18

```python
for i in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(i)
```


### Code Cell 19

```python
df.dtypes
```


### Code Cell 20

```python
df['journey_day'] = df['Date_of_Journey'].dt.day
```


### Code Cell 21

```python
df['journey_month'] = df['Date_of_Journey'].dt.month
```


### Code Cell 22

```python
df.drop(columns='Date_of_Journey',inplace=True)
```


### Code Cell 23

```python
df.head()
```


### Code Cell 24

```python
def extract_hour(data,col):
    data[col+'_hour']=data[col].dt.hour
    
def extract_min(data,col):
    data[col+'_min']=data[col].dt.minute
    

def drop_col(data,col):
    data.drop(col,axis=1,inplace=True)
```


### Code Cell 25

```python
#call the function
# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time
extract_hour(df,'Dep_Time')

#extracting minutes
extract_min(df,'Dep_Time')

#drop the column
drop_col(df,'Dep_Time')
df.head()
```


### Code Cell 26

```python
#extracting hour
extract_hour(df,'Arrival_Time')

#extracting min
extract_min(df,'Arrival_Time')


#drop the column
drop_col(df,'Arrival_Time')
df.head()
```


### Code Cell 27

```python
df['dur_hour'] = df['Duration'].str.extract(r'(\d+)h').fillna('0')
```


### Code Cell 28

```python
df['dur_min'] = df['Duration'].str.extract(r'(\d+)m').fillna('0')
```


### Code Cell 29

```python
df.sample(5)
```


### Code Cell 30

```python
df['dur_min'] = df['dur_min'].astype(int)
```


### Code Cell 31

```python
df['dur_hour'] = df['dur_hour'].astype(int)
```


### Code Cell 32

```python
df.info()
```


### Code Cell 33

```python
plt.figure(figsize = (25,18))
sns.boxplot(x = df.Airline, y = df.dur_hour, hue = df.Total_Stops)
plt.show()
```


### Code Cell 34

```python
sns.boxplot(df.dur_hour)
```


### Code Cell 35

```python
from scipy import stats
IQR = stats.iqr(df.dur_hour, interpolation = 'midpoint')
IQR
```


### Code Cell 36

```python
# First quartile (Q1) 
Q1 = df.dur_hour.quantile(0.25)

# Third quartile (Q3) 
Q3 =df.dur_hour.quantile(0.75)

##getting the limits
min_limit=Q1 - 1.5*IQR

max_limit=Q3 +1.5*IQR
print(min_limit)
print(max_limit)
```


### Code Cell 37

```python
import numpy as np
df.loc[df['dur_hour'] > max_limit,'dur_hour']=np.median(df.dur_hour)
```


### Code Cell 38

```python
sns.boxplot(df.dur_hour)
plt.show()
```


### Code Cell 39

```python
df.describe()
```


### Code Cell 40

```python
df[df['dur_hour'] == 0]
```


### Code Cell 41

```python
dg = df[df['Total_Stops']=='2 stops']
```


### Code Cell 42

```python
df['dur_hour'] = df['dur_hour'].replace(0, dg.dur_hour.mean())
```


### Code Cell 43

```python
df.describe()
```


### Code Cell 44

```python
sns.boxplot(df.dur_hour)
```


### Code Cell 45

```python
sns.boxplot(df.Price)
```


### Code Cell 46

```python
from scipy import stats
IQR = stats.iqr(df.Price, interpolation = 'midpoint')
IQR
# First quartile (Q1) 
Q1 = df.Price.quantile(0.25)

# Third quartile (Q3) 
Q3 =df.Price.quantile(0.75)

##getting the limits
min_limit=Q1 - 1.5*IQR

max_limit=Q3 +1.5*IQR
print(min_limit)
print(max_limit)
```


### Code Cell 47

```python
df.loc[df['Price'] > max_limit]
```


### Code Cell 48

```python
df.dtypes
```


### Code Cell 49

```python
df['dur_hour'] = df['dur_hour'].astype(int)
```


### Code Cell 50

```python
column = [col for col in df.columns if df[col].dtype == 'str']
column
```


### Code Cell 51

```python
continuous_col =[column for column in df.columns if df[column].dtype!='str']
continuous_col
```


### Code Cell 52

```python
categorical = df[column]
categorical.head()
```


### Code Cell 53

```python
categorical['Airline'].value_counts()
```


### Code Cell 54

```python
d2 = pd.get_dummies(df['Airline'],prefix = 'Airline',drop_first = True)
df = pd.concat([df,d2], axis = 1).drop(['Airline'],axis = 1)
#encoding of source 
d3 = pd.get_dummies(df['Source'],prefix = 'Source',drop_first = True)
df = pd.concat([df,d3], axis = 1).drop(['Source'],axis = 1)
#encoding of Destination
d4 = pd.get_dummies(df['Destination'],prefix = 'Destination',drop_first = True)
df = pd.concat([df,d4], axis = 1).drop(['Destination'],axis = 1)
#encoding of Total Stops
d5 = pd.get_dummies(df['Total_Stops'],prefix = 'Total_Stops',drop_first = True)
df = pd.concat([df,d5], axis = 1).drop(['Total_Stops'],axis = 1)
```


### Code Cell 55

```python
drop_col(df,'Additional_Info')
```


### Code Cell 56

```python
column=[column for column in df.columns if df[column].dtype=='str']
```


### Code Cell 57

```python
column
```


### Code Cell 58

```python
categorical['Route1'] = categorical['Route'].str.split('→').str[0]
categorical['Route2'] = categorical['Route'].str.split('→').str[1]
categorical['Route3'] = categorical['Route'].str.split('→').str[2]
categorical['Route4'] = categorical['Route'].str.split('→').str[3]
categorical['Route5'] = categorical['Route'].str.split('→').str[4]
```


### Code Cell 59

```python
categorical.head()
```


### Code Cell 60

```python
categorical.isnull().sum()
```


### Code Cell 61

```python
for i in ['Route3', 'Route4', 'Route5']:
    categorical[i] = categorical[i].fillna('None')
categorical.isnull().sum()
```


### Code Cell 62

```python
drop_col(categorical,'Route')
```


### Code Cell 63

```python
categorical.head()
```


### Code Cell 64

```python
from sklearn.preprocessing import LabelEncoder
```


### Code Cell 65

```python
encoder = LabelEncoder()
```


### Code Cell 66

```python
for i in ['Route1','Route2','Route3','Route4','Route5']:
    categorical[i] = encoder.fit_transform(categorical[i])
```


### Code Cell 67

```python
df = pd.concat([df,categorical], axis=1).drop(['Route'],axis=1)
```


### Code Cell 68

```python
df.describe()
```


### Code Cell 69

```python
df
```


### Code Cell 70

```python
df.drop(columns=['Duration','Source','Destination','Duration','Total_Stops','Additional_Info','Airline'],inplace=True)
```


### Code Cell 71

```python
df.head()
```


### Code Cell 72

```python
df.describe()
```


### Code Cell 73

```python
df.info()
```


### Code Cell 74

```python
df=df.astype(int)
```


### Code Cell 75

```python
df.info()
```


### Code Cell 76

```python
plt.figure(figsize=(30,30))
sns.heatmap(df.corr(),annot=True)
plt.plot()
```


### Code Cell 77

```python
x = df.iloc[:,1:]
y = df.loc[:,['Price']]
```


### Code Cell 78

```python
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(x, y)
```


### Code Cell 79

```python
ExtraTreesRegressor()
```


### Code Cell 80

```python
print(selection.feature_importances_)
```


### Code Cell 81

```python
feature_imp = pd.Series(selection.feature_importances_, index=x.columns).sort_values(ascending=False)
feature_imp
```


### Code Cell 82

```python
X = df[['Total_Stops_non-stop','journey_day','journey_month','Airline_Jet Airways','Airline_Jet Airways Business','dur_hour','Arrival_Time_hour','Route2','Dep_Time_hour','Route4','Airline_Multiple carriers','Dep_Time_min','Airline_IndiGo','Arrival_Time_min',                           'Destination_New Delhi',                        
'Route3',                                   
'dur_min']]
X
```


### Code Cell 83

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=3)
```


### Code Cell 84

```python
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
```


### Code Cell 85

```python
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
```


### Code Cell 86

```python
def predict(ml_model):
    print('Model is :{}'.format(ml_model))
    model = ml_model.fit(X_train,y_train)
    print("Training score {}".format(model.score(X_train,y_train)))
    predictions = model.predict(X_test)
    print("Predictions are : {}".format(predictions))
    print('\n')
    r2score = r2_score(y_test,predictions)
    print("r2 score is : {}".format(r2score))

    print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
    print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
```


### Code Cell 87

```python
predict(LinearRegression())
```


### Code Cell 88

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.6069)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 89

```python
predict(RandomForestRegressor())
```


### Code Cell 90

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.8015)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 91

```python
predict(DecisionTreeRegressor())
```


### Code Cell 92

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.7224)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 93

```python
predict(GradientBoostingRegressor())
```


### Code Cell 94

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.7836)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 95

```python
from xgboost import XGBRegressor
predict(XGBRegressor())
```


### Code Cell 96

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.8219)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 97

```python
# Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV
## Number of trees in ramdom forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
## Number of features to consider at every split
max_features = ['auto', 'sqrt']
## Maximum number of level in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
## Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
## Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
```


### Code Cell 98

```python
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)
```


### Code Cell 99

```python
rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
```


### Code Cell 100

```python
predict(rf_random)
```


### Code Cell 101

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.80615)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 102

```python
random_grid = {
    'n_estimators' : [100,300,500,700,900,1100],
    'max_depth':[5,20,35,50],
    }
xgb=XGBRegressor()
xg_random=RandomizedSearchCV(estimator=xgb,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1,)
```


### Code Cell 103

```python
predict(xg_random)
```


### Code Cell 104

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.8335)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 105

```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
```


### Code Cell 106

```python
predict(grid_search)
```


### Code Cell 107

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.7851)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


### Code Cell 108

```python
grid_search.best_params_
```


### Code Cell 109

```python
#XGBoost hyper-parameter tuning
param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }
```


### Code Cell 110

```python
xgb_model = XGBRegressor()
gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)
```


### Code Cell 111

```python
predict(gsearch)
```


### Code Cell 112

```python
# Adjusted r2 = 1-(1-R2)*(n-1)/(n-p-1)
adjusted_r2_score = 1-(1-0.8398)*(2671-1)/(2671-18-1)
adjusted_r2_score
```


</details>

## Resume Project Title
**Flight Fare Prediction Using Machine Learning and Feature Engineering**

**Resume description:** Built a flight fare prediction model using exploratory analysis, date/time feature extraction, outlier treatment, categorical encoding, feature selection, and regression algorithms; tuned models and evaluated them using R², MAE, MSE, and RMSE.
