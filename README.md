# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load dataset and libraries.

2. Split data into features and labels, then into train and test sets.

3. Train the Linear Regression model.

4. Predict using the test data.

5. Evaluate performance and visualize results. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ABISHA RANI S
RegisterNumber: 212224040012

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')
print(dataset.head())
dataset=pd.read_csv('data.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
 
*/
```

## Output:

HEAD VALUE

<img width="191" height="131" alt="image" src="https://github.com/user-attachments/assets/3f7b203c-a3ae-485d-b119-290ac23e3c9d" />

TAIL VALUES

<img width="203" height="125" alt="image" src="https://github.com/user-attachments/assets/beefde4d-6867-4db3-9573-bd7fa16b3518" />

COMPARE DATASET

<img width="588" height="473" alt="image" src="https://github.com/user-attachments/assets/9c5af3a1-13e4-44a4-a7f6-93595d42065a" />

PREDICTED X AND Y

<img width="650" height="66" alt="image" src="https://github.com/user-attachments/assets/abab3d61-3e85-4b74-9e30-c387aae9436c" />

TRAINING SET

<img width="640" height="516" alt="image" src="https://github.com/user-attachments/assets/cb666f04-8940-4b11-bce3-76ad4e4a22f1" />

TESTING SET

<img width="625" height="508" alt="image" src="https://github.com/user-attachments/assets/7c51c2ab-1f99-4294-ad21-d0d8022dd609" />

MSE,MAE,RMSE

<img width="412" height="90" alt="image" src="https://github.com/user-attachments/assets/02d59599-ae32-49e7-89fc-754b5e5dfb8f" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
