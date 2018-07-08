# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #All columns except the last one
y = dataset.iloc[:, 3].values # all values of last column (3)

# Taking care of missing data
from sklearn.preprocessing import Imputer
#imputer class helps us to take care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #0 = columns. 1=rows
imputer = imputer.fit(X[:, 1:3]) #fitting imputer to x's 1 and 2 column
X[:, 1:3] = imputer.transform(X[:, 1:3]) #transform will apply mean strategy and will update x

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #encoding categorical variable german,france,spain as 1,2,3
onehotencoder = OneHotEncoder(categorical_features = [0]) #encoding dummy variables
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the Dataset into the training set and test set

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

# x_train and x_test are scaled in same range because we fit StandardScaler object to x_train first

x_test = sc_x.transform(x_test)