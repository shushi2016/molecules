#!/usr/bin/env python
#! coding: utf-8

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import ipdb
from utils import customeMAE

train = pd.read_pickle('../01_data/myTrain.pkl')

train_dummy = pd.get_dummies(train[['atom_index_0',
                                   'atom_index_1',
                                   'type']])
Y = train.scalar_coupling_constant

X_train = train_dummy[train.group=='TRAIN']
Y_train = Y[train.group=='TRAIN']

X_test = train_dummy[train.group=='TEST']
Y_test = Y[train.group=='TEST']

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

mdl = GradientBoostingRegressor()
#mdl.fit(X_train, Y_train)

#Y_pred = mdl.predict(X_test)
#print("MAE = {}".format(customeMAE(train[train.group=='TEST']['type'],
#                                   Y_test,
#                                   Y_pred)))

# fit all data and predict on real test.csv
test = pd.read_csv('../01_data/test.csv')
ipdb.set_trace()
mdl.fit(train_dummy, Y)
test_dummy = pd.get_dummies(test[['atom_index_0',
                                  'atom_index_1',
                                  'type']])
pred = mdl.predict(test_dummy)

submission = pd.DataFrame({'id': test.id,
                           'scalar_coupling_constant': pred})
submission.to_csv('simple_baseline_20190603.csv',
                    index=False)
