### Implementation-of-Logistic-Regression-Using-Gradient-Descent

### AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
1.Import the required libraries.
2.Load the dataset and print the values.
3.Define X and Y array and display the value.
4.Find the value for cost and gradient.
5.Plot the decision boundary and predict the Regression value. 

### Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JABEZ S
RegisterNumber: 212223040070

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
*/
```

### Output:
## Dataset:
![Screenshot 2024-09-13 190128](https://github.com/user-attachments/assets/2a8b62d9-1cb0-46f5-b0d2-bc7fe08f7dd7)
![Screenshot 2024-09-13 190208](https://github.com/user-attachments/assets/cd24824d-3d6e-42ab-8f9e-81fb40adfff3)
![Screenshot 2024-09-13 190222](https://github.com/user-attachments/assets/f775e873-05ab-452b-90c6-def0527faecb)
![Screenshot 2024-09-13 190232](https://github.com/user-attachments/assets/32899843-11d3-4f85-8bb4-a555c3d1ce0e)

## Accuracy and Predicted Values:
![Screenshot 2024-09-13 190252](https://github.com/user-attachments/assets/22b13f68-b4c0-4e21-b34e-e1ecce1c7f62)
![Screenshot 2024-09-13 190302](https://github.com/user-attachments/assets/a50aa09c-e9d1-4583-bf40-4f8dd19f3abe)
![Screenshot 2024-09-13 190311](https://github.com/user-attachments/assets/3927f70d-56d4-4615-a8a5-18003a2a61cc)
![Screenshot 2024-09-13 190331](https://github.com/user-attachments/assets/021de21d-d570-4c69-bd28-189d609a3011)





### Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

