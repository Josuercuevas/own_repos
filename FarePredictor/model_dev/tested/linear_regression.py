from pickle import FALSE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from os.path import join

'''
Trip ID
Taxi ID
Trip Seconds
Trip Miles
Pickup Census Tract
Dropoff Census Tract
Pickup Community Area
Dropoff Community Area
Fare
Tips
Tolls
Extras
Payment Type
Company


==== Pred ====:
Fare
'''
filename = join(join(join(join('..', '..'), 'dataset'), 'chicago_noprivacy'), 'dataset.csv')
fulldb = pd.read_csv(filename, sep=',')


all_features = fulldb.drop('Fare', axis=1)
all_targets = fulldb.drop(fulldb.columns.difference(['Trip ID', 'Fare']), axis=1)
print(all_features.columns.to_list())
print(all_targets.columns.to_list())

ENABLE_TAXIID = False
if ENABLE_TAXIID:
    # Taxi ID is categorical, need to do one-hot-encoding
    TaxiId = pd.get_dummies(all_features['Taxi ID'], drop_first=True)
    all_features = pd.concat([all_features, TaxiId],axis=1)
    all_features = all_features.drop('Taxi ID', axis=1)
else:
    all_features = all_features.drop('Taxi ID', axis=1)

# Payment Type is categorical, need to do one-hot-encoding
PaymentType = pd.get_dummies(all_features['Payment Type'], drop_first=True)
all_features = pd.concat([all_features, PaymentType],axis=1)
all_features = all_features.drop('Payment Type', axis=1)

# Company is categorical, need to do one-hot-encoding
Company = pd.get_dummies(all_features['Company'], drop_first=True)
all_features = pd.concat([all_features, Company],axis=1)
all_features = all_features.drop('Company', axis=1)

print(f'Total number of dimensions to deal with: {len(all_features.columns.to_list())}')

# Trip Seconds
'''No need to do anything this variable is not categorical'''
# Trip Miles
'''No need to do anything this variable is not categorical'''

ENABLE_CENSUS_TRACT = False
if ENABLE_CENSUS_TRACT:
    # Dropoff Census Tract
    '''No need to do anything this variable is not categorical'''
    # Pickup Census Tract
    '''No need to do anything this variable is not categorical'''
else:
    # remove it is not needed
    all_features = all_features.drop(['Pickup Census Tract', 'Dropoff Census Tract'], axis=1)

ENABLE_COMMUNITY_AREA = False
if ENABLE_COMMUNITY_AREA:
    # Pickup Community Area
    '''No need to do anything this variable is not categorical'''
    # Dropoff Community Area
    '''No need to do anything this variable is not categorical'''
else:
    # remove it is not needed
    all_features = all_features.drop(['Pickup Community Area', 'Dropoff Community Area'], axis=1)

ENABLE_TIPS_TOLLS_EXTRAS = False
if ENABLE_TIPS_TOLLS_EXTRAS:
    # Tips
    '''No need to do anything this variable is not categorical'''
    # Tolls
    '''No need to do anything this variable is not categorical'''
    # Extras
    '''No need to do anything this variable is not categorical'''
else:
    all_features = all_features.drop(['Tips', 'Tolls', 'Extras'], axis=1)

Y = all_targets.drop('Trip ID', axis=1)
X = all_features.drop('Trip ID', axis=1)

print('==================')
print(X.columns.to_list())
print(Y.columns.to_list())

print(Y.head())
print(X.head())

# split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
LR = LinearRegression()
regressor = LR.fit(x_train, y_train)
print(regressor.coef_)

# now lets do some predictions
y_prediction = regressor.predict(x_test)
score = r2_score(y_test, y_prediction)
print(y_prediction[0:3])
print(y_test[0:3])
print(f'r2 score is: {score}')
print(f'Mean square error is: {mean_squared_error(y_test, y_prediction)}')
print(f'Root mean square error is: {np.sqrt(mean_squared_error(y_test, y_prediction))}')