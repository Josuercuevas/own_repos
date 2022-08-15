import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

'''
Trip ID
Taxi ID
Trip Seconds
Trip Miles
Fare
Tips
Tolls
Extras
Payment Type
Company


==== Pred ====:
Fare
'''
filename = join(join(join(join('..', '..'), 'dataset'), 'pooled'), 'dataset.csv')
fulldb = pd.read_csv(filename, sep=',')

all_features = fulldb.drop('Fare', axis=1)
all_targets = fulldb.drop(fulldb.columns.difference(['Trip ID', 'Fare']), axis=1)
print(all_features.columns.to_list())
print(all_targets.columns.to_list())


ENABLE_TAXIID = False
if ENABLE_TAXIID:
    # Taxi ID is categorical, need to do one-hot-encoding
    le = LabelEncoder()
    all_features['Taxi ID'] = le.fit_transform(all_features['Taxi ID'])
else:
    all_features = all_features.drop('Taxi ID', axis=1)

# Payment Type is categorical, need to do one-hot-encoding
le = LabelEncoder()
all_features['Payment Type'] = le.fit_transform(all_features['Payment Type'])

# Company is categorical, need to do one-hot-encoding
le = LabelEncoder()
all_features['Company'] = le.fit_transform(all_features['Company'])

# Trip Seconds
'''No need to do anything this variable is not categorical'''
# Trip Miles
'''No need to do anything this variable is not categorical'''

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
print(f'Number of samples pooled Training: {len(x_train)}')
print(f'Number of samples pooled Testing: {len(x_test)}')
XGBoost = XGBRegressor(n_estimators=200, learning_rate=0.1, n_jobs=8)
regressor = XGBoost.fit(x_train, y_train.values.ravel())

# now lets do some predictions
y_prediction = regressor.predict(x_test)
score = r2_score(y_test, y_prediction)
print(y_prediction[0:3])
print(y_test[0:3])
importances = regressor.feature_importances_
print(f'Feature importances are: {importances}')
print(f'r2 score is: {score}')
print(f'Mean square error is: {mean_squared_error(y_test,y_prediction)}')
print(f'Root mean square error is: {np.sqrt(mean_squared_error(y_test,y_prediction))}')

indices = np.argsort(importances)
fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(x_train.columns)[indices])
plt.xlabel("Importance weight")
plt.ylabel("Feature Name")
plt.title("Feature Importance Plot")
plt.savefig('Feature_Importance.png')
plt.close('all')

plt.scatter(all_features['Trip Seconds'][0:1000], all_targets['Fare'][0:1000])
plt.xlabel("Trip Seconds")
plt.ylabel("Fare")
plt.title("Trip Seconds vs Fare Relationship")
plt.xticks(rotation='vertical')
plt.savefig('tripseconds_fare_relationship.png')
plt.close('all')

plt.scatter(all_features['Trip Miles'][0:1000], all_targets['Fare'][0:1000])
plt.xlabel("Trip Miles")
plt.ylabel("Fare")
plt.title("Trip Miles vs Fare Relationship")
plt.xticks(rotation='vertical')
plt.savefig('tripmiles_fare_relationship.png')
plt.close('all')
