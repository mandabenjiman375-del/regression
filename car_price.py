import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
data = pd.read_csv(r"C:\Users\Lenovo\Downloads\archive (4)\used_cars.csv")
print(data)
print(data.info())
print(data.isnull().sum())

#filling missing values
data['fuel_type']=data['fuel_type'].fillna(data['fuel_type'].mode()[0])
data['accident']=data['accident'].fillna(data['accident'].mode()[0])
data['clean_title']=data['clean_title'].fillna(data['clean_title'].mode()[0])

print(data.isnull().sum())

#price convert object to numeric
data['price']=data['price'].str.replace('$','').str.replace(',','').astype(int)
data['milage']=data['milage'].str.replace('mi','').str.replace(',','').str.replace('.','').astype(int)
# Engine liters
data['engine_liters'] = data['engine'].str.extract('([\d\.]+)L').astype(float)
data['engine_liters']=data['engine_liters'].fillna(data['engine_liters'].mean())

# Horsepower
data['horsepower'] = data['engine'].str.extract('([\d\.]+)HP').astype(float)
data['horsepower']=data['horsepower'].fillna(data['horsepower'].mean())

# Electric car flag
data['is_electric'] = data['engine'].str.contains('Electric', case=False, na=False).astype(int)

# Drop original engine column
print(data.info())
print(data.isnull().sum())


#clean binary columns
data['accident']=data['accident'].map({'At least 1 accident or damage reported':1,' None reported':0})
data['accident']=data['accident'].fillna(data['accident'].mean())
print(data['accident'])

data['clean_title']=data['clean_title'].map({'Yes':1})
print(data['clean_title'].unique())

#drop model because it has too many unique values
print(data.drop('model',axis=1,inplace=True))

#checking outliers
print(plt.figure(figsize=(10,10)))
print(sns.boxplot(data))
print(plt.show())

#remove skew
print(data['price'].skew())
data['price']=np.log1p(data['price'])
print(data['price'].skew())

print(data['milage'].skew())
data['milage']=np.log1p(data['milage'])
print(data['milage'].skew())

print(plt.figure(figsize=(10,10)))
print(sns.boxplot(data))
print(plt.show())

data['horsepower']=np.log1p(data['horsepower'])

print(plt.figure(figsize=(10,10)))
print(sns.boxplot(data))
print(plt.show())

print(data.info())
print(data.isnull().sum())

#data encodeing for categorical columns
cat_cols=['brand','ext_col','int_col','brand','fuel_type','transmission']
le=LabelEncoder()
for col in cat_cols:
    data[col]=le.fit_transform(data[col])

#drop original engine
print(data.drop('engine',axis=1,inplace=True))    

print(data.head())

#split feature
x=data.drop('price',axis=1)
y=data['price']

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

#train the model
model=LinearRegression()
print(model.fit(x_train,y_train))

#evaluate
y_pred=model.predict(x_test)

#checking metrics
print('r2:',r2_score(y_pred,y_test))
print('MSE:',mean_squared_error(y_pred,y_test))

#train model using randomforest regressor
model = RandomForestRegressor(random_state=42,n_estimators=300,max_depth=None,min_samples_leaf=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))



pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))