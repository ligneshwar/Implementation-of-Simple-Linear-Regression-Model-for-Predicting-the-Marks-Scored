# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K.Ligneshwar
RegisterNumber:  212223230113
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/7f0f7403-ea1c-4c3b-9fa3-7b49d6f098e0)
![image](https://github.com/user-attachments/assets/6f2f943a-e0c6-4c2e-8ff4-560554a8b241)
![image](https://github.com/user-attachments/assets/8a8a42b4-82f9-43c0-a25d-74864848c78c)
![image](https://github.com/user-attachments/assets/2c83bdf6-42a4-4e0a-a733-4e05a7923875)
![image](https://github.com/user-attachments/assets/60d52e80-d7b1-4d7e-a200-4213e927c389)
![image](https://github.com/user-attachments/assets/6308ddb5-2224-408b-b1bc-cc24fa3c3db6)
![image](https://github.com/user-attachments/assets/11d8cf67-d698-46d1-83b0-437b0dfb78ee)
![image](https://github.com/user-attachments/assets/d972f881-5443-4063-9c3d-ed47d4244e11)
![image](https://github.com/user-attachments/assets/99baa0ba-75da-4dee-b08f-eee4bad4d7d0)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
