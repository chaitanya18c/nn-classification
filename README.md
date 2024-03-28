# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![image](https://github.com/chaitanya18c/nn-classification/assets/119392724/c0faf9e0-d57a-4a45-9ca5-f3de7e9ba8c5)

## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules

### STEP 2:
Load and clean the customer data, handling missing values.

### STEP 3:
Use OrdinalEncoder and LabelEncoder to encode categorical features.

### STEP 4:
Generate a correlation matrix and heatmap, along with a pair plot for data exploration.

### STEP 5:
Apply MinMax scaling to the 'Age' column in the dataset.

### STEP 6:
Create a neural network model using TensorFlow/Keras with specified layers and activation functions.

### STEP 7:
Train the model on the scaled training data with early stopping for preventing overfitting.

### STEP 8:
Evaluate the trained model on the test data, save the model, and store necessary data for future predictions.


## PROGRAM

### Name: CHAITANYA P S 
### Register Number: 212222230024

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv('customers.csv')
df.head()
df.columns
df.dtypes
df.shape
df.isnull().sum()
df_cleaned=df.dropna(axis=0)
df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()
df_cleaned['Graduated'].unique()
df_cleaned['Family_Size'].unique()
df_cleaned['Var_1'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Segmentation'].unique()

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categories_list=[['Male','Female'],
                 ['No','Yes'],
                 ['No','Yes'],
                 ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
                 'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                 ['Low','Average','High']
                 ]
enc = OrdinalEncoder(categories = categories_list)

customer1=df_cleaned.copy()
customer1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]=enc.fit_transform(customer1[
                                                                                ['Gender','Ever_Married','Graduated','Profession','Spending_Score']
                                                                             ])

customer1.dtypes
le = LabelEncoder()
customer1['Segmentation'] = le.fit_transform(customer1['Segmentation'])
customer1.dtypes
customer1 = customer1.drop('ID',axis=1)
customer1 = customer1.drop('Var_1',axis=1)
     
customer1.dtypes
customer1.describe()
customer1['Segmentation'].unique()

X=customer1[['Gender','Ever_Married','Age','Graduated',
          'Profession','Work_Experience',
          'Spending_Score','Family_Size']].values

y1 = customer1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray() 
y.shape    
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

# Creating the model
ai_brain = Sequential([
  Dense(4,input_shape=(8,)),
  Dense(8,activation='relu'),
  Dense(8,activation='relu'),
  Dense(4,activation='softmax'),
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)


ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
 
# Saving the Model
ai_brain.save('customer_classification_model.h5')

# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customer1,df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

ai_brain = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
     
#Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))


```

## Dataset Information
![310541491-e33ecbfe-7925-40a6-9662-09b957734f1d](https://github.com/chaitanya18c/nn-classification/assets/119392724/ce2c314e-84b5-4be2-9822-a776783588ab)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![download](https://github.com/chaitanya18c/nn-classification/assets/119392724/d992c020-0bc5-4246-85ca-2b3c0facf168)


### Classification Report
![310541696-1e32082e-499d-4981-a652-1131e42d430a](https://github.com/chaitanya18c/nn-classification/assets/119392724/1b95a703-d552-4c5a-96eb-07d74932de71)

### Confusion Matrix
![310542136-d8c3692b-8733-4d3a-b6cd-4019c38a8643](https://github.com/chaitanya18c/nn-classification/assets/119392724/3ae7340f-87a2-489b-87f9-8de8f2cf94d7)


### New Sample Data Prediction
![310542006-02b0fd9c-e37b-4af9-81d8-497d36e15f2e](https://github.com/chaitanya18c/nn-classification/assets/119392724/055799db-2f42-4439-b5d8-baae480a9772)

## RESULT
Thus a neural network classification model is developed for the given dataset.
