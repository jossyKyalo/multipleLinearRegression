import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
from sklearn.compose import ColumnTransformer

data_set=pd.read_csv('50_Startups.csv')

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 4].values  
print(x)
print("\n")
print(y)
#handling categorical data using ColumnTransformer and OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],  # Apply OneHotEncoder to the 4th column (index 3)
    remainder='passthrough' 
)

x = column_transformer.fit_transform(x)

#avoiding the dummy variable trap:  
x = x[:, 1:]  

print("\nTransformed x:\n",x)   

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0) 

 #Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  

#Predicting the Test set result;  
y_pred= regressor.predict(x_test)
print(y_pred)

#score for training dataset and test dataset
print('\nTrain Score: ', regressor.score(x_train, y_train))  
print('\nTest Score: ', regressor.score(x_test, y_test))  