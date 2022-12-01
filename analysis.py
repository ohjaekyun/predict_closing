from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torch import tensor as Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
import random
from models.rnn import CMV_LSTM, MV_LSTM
from models.vae import ConditionalVAE, VanillaVAEEncoder, VanillaVAEDecoder
from models.rvae import UnitRVAE, RVAE
from utils.loss import VAE_Loss
from catboost import CatBoostClassifier, CatBoostRegressor


def get_area_under_lorenz(y, y_pred):
    n = len(y_pred)
    m = len(y.nonzero()[0])
    width = n - m
    sort_idx = np.argsort((-1) * y_pred)
    sort_y = y[sort_idx]

    #For calculating K-S
    ks_stat = 0
    grate = 0
    brate = 0
    area = 0
    for i, yv in enumerate(sort_y):
        if yv == 1:
            area += width#(n - i - 0.5) / (n * m)
            brate += 1 / m
            ks_stat = max(ks_stat, brate - grate)
        else:
            width -= 1
            grate += 1 / (n - m)

    aulc = area / ((n - m) * m)
    print(f"K-S stat: {ks_stat}")
    print(f"GINI: {2 * (aulc - 0.5)}")
    return aulc


def predict_forest(models, x):
    preds = 0
    for model in models:
        preds += model.predict_proba(x)[:, 1]
    return preds / len(models)


def importance_forest(models):
    importances = 0
    for model in models:
        importances += model.get_feature_importance()
    return importances / len(models)


data = pd.read_csv('data/data_for_predict.csv', index_col=0)
y = data['label'].to_numpy()
data = data.drop(columns=['BIZ_NO', 'label'])
cols = data.columns
list_cols = list(cols)
categorical_features_names = ['BZ_TYP',]
categorical_features_idxs = []
for catname in categorical_features_names:
    try:
        categorical_features_idxs.append(list_cols.index(catname))
    except:
        pass

x_remain, xtest, y_remain, ytest = train_test_split(data, y, train_size = 0.9, random_state = 179)
models = []
random_states = [42, 1, 917, 7, 43, 68, 50, 111, 456, 12, 34, 25, 234, 37, 690, 2459, 89, 347, 1004, 829]
for random_state in random_states:
    xtrain, xvalidation, ytrain, yvalidation = train_test_split(x_remain, y_remain, train_size = 0.8, random_state = random_state)

    params_init = {'iterations' : 30000, 'learning_rate' : 0.1, 'custom_loss' : ['AUC', 'Accuracy'], 'use_best_model' : False, 'sampling_frequency' : 'PerTreeLevel', 
    'max_depth' : 8, 'model_shrink_mode' : 'Decreasing', 'model_shrink_rate' : 0.995, 'grow_policy' : 'SymmetricTree', 'l2_leaf_reg' : 50, 'random_strength' : 16, 
    'border_count' : 80, 'bagging_temperature' : 0.02, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'metric_period': 100}
    params_init['custom_loss'].append('Logloss')
    if params_init["bootstrap_type"] == "Bernoulli":
        params_init["bagging_temperature"] = None
    #bootstrap_type: Bayesian, Bernoulli, MVS
    params_train = {'use_best_model': True, 'early_stopping_rounds': 1500, 'cat_features': categorical_features_idxs}
    model = CatBoostClassifier(**params_init)
    model.fit(xtrain, ytrain, eval_set = (xvalidation, yvalidation), **params_train)
    preds_train = model.predict_proba(xtrain)[:, 1]
    preds_test = model.predict_proba(xtest)[:, 1]
    preds_val = model.predict_proba(xvalidation)[:, 1]
    models.append(model)
    aulc = get_area_under_lorenz(ytest, preds_test)
    from catboost import Pool
    params_prediction = {'cat_features' : categorical_features_idxs}
    testpool = Pool(xtest, ytest, **params_prediction)
    met_test = model.eval_metrics(testpool, metrics = params_init['custom_loss'])
    for k in list(met_test.keys()):
        print(f"Test {k}: ", met_test[k][-1])
    feature_importances = model.get_feature_importance()

preds_test = predict_forest(models, xtest)
aulc = get_area_under_lorenz(ytest, preds_test)
importances = importance_forest(models)

x = 1