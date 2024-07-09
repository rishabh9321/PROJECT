

import pandas as pd
import numpy as np
import seaborn as sns

# To partition the dat 
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# Importing Data
data_income = pd.read_csv("D:\income.csv" )

# Creating a copy of original data                                                                              
data = data_income.copy()

#  read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
data = pd.read_csv("D:\income.csv",na_values=[" ?"]) 

# Creating a copy of original data                                                                              
data = data_income.copy()


#  read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
data = pd.read_csv("D:\income.csv",na_values=[" ?"]) 


data2 = data.dropna(axis=0)

# The above code I have done already  in deep  in the python file "1 . classification .ipynb"




# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)


# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

"""From the confusion_matrix output >>>>>>  if the actual class is less than or
equal to 50000 and the model has predicted 6332 observation as less than or 
equal 50000 ,  But being less than or equal to 50000 is the actual class ,
the model has predicted 491 observation as greater than 50000 .
          similarly given the actual salary status is greater than 50000,the
          model has predicted 1301 observation as greater than 50000  and 925
          observation as less than or equal to 50000
So, there are many misclassification is here .so, modelhas not classified al
the observation correctly .so using a measure called accuracy we will be able
to get the accuracy score of the model that variable """



# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

""" From the abobe line code >>>> Accuracy score is 0.8366670. so it means that 
83% of the time the model is able to classify the record correctly """


# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

"""Misclassified sample output  is 1416"""



"""YOU CAN ALSO IMPROVE THE ACCURACY OF THE MODEL BY REDUCEING THE NUMBER OF 
MISCLASSIFIED SAMPLE . SO ONE OF THE METHOD IS BY REMOVING THE INSIGNIFICANT
VARIABLES
 """

# =============================================================================
# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns

# To partition the dat 
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

# Importing Data
data_income = pd.read_csv("D:\income.csv" )

# Creating a copy of original data                                                                              
data = data_income.copy()

#  read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
data = pd.read_csv("D:\income.csv",na_values=[" ?"]) 

# Creating a copy of original data                                                                              
data = data_income.copy()


#  read the data by including "na_values[' ?']" to consider ' ?' as nan !!!
data = pd.read_csv("D:\income.csv",na_values=[" ?"]) 


data2 = data.dropna(axis=0)

# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data2.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features=list(set(columns_list2)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)

# Prediction from test data
prediction = logistic.predict(test_x)

#Calculating the accuracy score
accuracy_score2=accuracy_score(test_y, prediction)
print(accuracy_score)

"""Now accuracy turned out to be 0.834788. the accuracy has a little bit 
droped down from above acuracy , because here we have removed all the 
insignificane variables .
            so, by removing insignificance variables we are not getting
            a better model which gives us better accuracy. when we remove
            any variable , ofcourse we are losing some information from the
            data ,that's why we are getting a decrease in accuracy """



# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y!= prediction).sum())

"""Misclassified samples: 1495"""










# =============================================================================
# KNN
# =============================================================================



# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier  


# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)  

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

# Performance metric check
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

""" Confusion matric output =>[[6338,485],
                               [941,1285]]    """




# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

""" Accuracy score is 0.83534092"""



print('Misclassified samples: %d' % (test_y != prediction).sum())

""" Misclassified samples: 1490"""


"""
Effect of K value on classifier
"""
Misclassified_sample = []
# Calculating error for K values between 1 and 20
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)

''' 
MISSCLASSIFIED_SAMPLE'S OUTPUT 

[1723, 1512, 1541, 1480, 1490, 1442, 1467, 1433, 1437, 1416, 1413, 1434, 1437,
 1445, 1426, 1420, 1438, 1423, 1441]    

  '''
  
''' CONCLUSION =>  There are 16th value of K is  1413 which is less missclassified sample ,then we should 
take K=16 '''




#  .................CONCLUSION (Which model is better)............

'''Logistic Regression and the other one is KNN bothof the algorithams gives same
  performance when we looked at the accuracy and in terms of misclassification.
  But on the whole , The the KNN is formining a little bitbetter with the 
  accuracy of 0.83534090    ''' 


