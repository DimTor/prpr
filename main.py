import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

train = pd.read_csv('train_final.csv')
test = pd.read_csv('test_final.csv')
train = train.drop(columns=['reservation_status_date', 'country', 'assigned_room_type', 'reserved_room_type'])
test = test.drop(columns=['reservation_status_date', 'country', 'assigned_room_type', 'reserved_room_type'])
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(train['arrival_date_month'])
train['arrival_date_month'] = label_encoder.transform(train['arrival_date_month'])
test['arrival_date_month'] = label_encoder.transform(test['arrival_date_month'])
categ = ['market_segment', 'customer_type', 'meal', 'distribution_channel']
train = pd.get_dummies(train, columns=['market_segment', 'customer_type', 'meal', 'distribution_channel'])
test = pd.get_dummies(test, columns=['market_segment', 'customer_type', 'meal', 'distribution_channel'])
train = pd.get_dummies(train, columns=['hotel', 'market_segment', 'deposit_type'])
test = pd.get_dummies(test, columns=['hotel', 'market_segment', 'deposit_type'])
plt.bar(train['meal'].unique(), train['meal'].value_counts())
plt.show()