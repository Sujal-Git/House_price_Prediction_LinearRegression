#Importing Libraries
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df=pd.read_csv('housing.csv')

#Removing  irrelevant columns from the dataframe
data=df.drop(['waterfront','view','yr_renovated'],axis=1)
x=data.iloc[:,1:19].values
y=data.iloc[:,0].values

#Spliting the data into test and training
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=3)

#Feeding data to model
Model=linear_model.LinearRegression()
Model.fit(x_train,y_train)

#Testing the model
predicted=Model.predict([[3,2.25,2570,7242,2.0,3,7,2170,400,1951,98125,47.7210,-122.319,1690,7639]])
print(f"Predicted Price of House is: {predicted}")

##Coder - Sujal Vishwakarma


