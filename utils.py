#!/usr/bin/env python
#1 coding: utf-8

from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


def customeMAE(y_type, y_truth, y_predict):
    df = pd.DataFrame({'y_type': y_type,
                      'y_truth': y_truth,
                      'y_predict': y_predict})
    dfg = df.groupby('y_type').apply(
            lambda x: mean_absolute_error(x['y_truth'], x['y_predict']))
    dfg = dfg.clip_lower(1e-9)
    log_dfg = np.log(dfg)
    return log_dfg.mean()
