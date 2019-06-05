#!/usr/bin/env python
#! coding: utf-8
import pandas as pd
from utils import customeMAE
import random
random.seed(123)
import swifter

train = pd.read_csv('../01_data/train.csv')

all_mol_names = train.molecule_name.unique()
random.shuffle(all_mol_names)

mol_names_test = set(all_mol_names[0:8500])
mol_names_valid = set(all_mol_names[8500:17000 ])
mol_names_train = set(all_mol_names[17000:])


def assign_group(name):
    if name in mol_names_test:
        return 'TEST'
    elif name in mol_names_train:
        return 'TRAIN'
    elif name in mol_names_valid:
        return 'VALID'
    else:
        return 'SomethingWrong'

train['group'] = train.molecule_name.swifter.apply(
        lambda x: assign_group(x))

assert train.group.nunique()==3

train.to_pickle('../01_data/myTrain.pkl')
