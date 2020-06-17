#Classification Model Based on Logistic Regression
#Model to Calculate if patient would Die of Heart Failure
#Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#creating dataset

dataset = pd.read_csv("heart_multivariate.csv")
X_data = dataset.iloc[:,:11]
Y_data = dataset.iloc[:,12]
print(dataset.head())

#creating dataset

dataset = pd.read_csv("heart_multivariate.csv")
X_data = dataset.iloc[:,:11]
Y_data = dataset.iloc[:,12]
print(dataset.head())

#creating dataset

dataset = pd.read_csv("heart_multivariate.csv")
X_data = dataset.iloc[:,:11]
Y_data = dataset.iloc[:,12]
print(dataset.head())

#fitting data & applying regression

LogReg = LogisticRegression()
LogReg.fit(X_train , Y_train)
Y_pred = LogReg.predict(X_test)

#confusion Matrix and Classification Report with accuracy for death report

CM1 = confusion_matrix(Y_test , Y_pred)
CR1 = classification_report(Y_test , Y_pred)
AC1 = accuracy_score(Y_test , Y_pred)
print ("FOR DEATH REPORT")
print("confusion Matrix --")
print(CM1)
print("------------------------------------------")
print("\nClassification Report--")
print(CR1)
print("------------------------------------------")
print("\nAccuracy Score--")
print(AC1)
print("------------------------------------------")

#perdicting death on external data

Ex_data = np.array([78.0, 0 ,582, 1 ,20,1 , 265000.00 , 1.9, 130,1, 1 ])
Y_ex_data_pred = LogReg.predict(Ex_data.reshape(1,-11))
if (Y_ex_data_pred == [1]):
    print("Death is Expected")
else:
    print ("Death Unexpected")
    
#Predicting data by user entry

Age=float(input("Enter patient's age -- "))
Ane=float(input("Is the patient anemic\n \tEnter 1 for Yes and 0 for No--  "))
Ce_prt=float(input('Enter creatinine_phosphokinase level--'))
Dia=float(input("Is the patient diabetic\n \tEnter 1 for Yes and 0 for No--  "))
Ej_Fr = float(input('Enter ejection_fraction --'))
HBP=float(input("Does the patient have high BP\n \tEnter 1 for Yes and 0 for No--  "))
plt=float(input("Enter platelet count--  "))
Se_ce=float(input("Enter serum_creatinine level--  "))
Se_So=float(input("Enter serum_sodium level--  "))
sex=float(input("Enter sex of the patient\n \tEnter 1 for Male and 0 for Female--  "))
smo=float(input("Does the patient smoke\n \tEnter 1 for Yes and 0 for No--  "))
User_data=np.array([Age , Ane , Ce_prt , Dia , Ej_Fr , HBP , plt , Se_ce, Se_So, sex , smo])
Y_user_data_pred= LogReg.predict(User_data.reshape(1,-11))
if (Y_user_data_pred == [1]):
    print("Death is Expected")
else:
    print ("Death Unexpected")
