import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from lightgbm import log_evaluation

INPUT_DIR = "/Users/tatsuyama4723/Desktop/GCI/"

train = pd.read_csv(INPUT_DIR + "train.csv")
test = pd.read_csv(INPUT_DIR + "test.csv")
sample_sub = pd.read_csv(INPUT_DIR + "sample_submission.csv")

use_features = [ "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3","DAYS_BIRTH","DAYS_LAST_PHONE_CHANGE",
            "DAYS_ID_PUBLISH","AMT_GOODS_PRICE","DAYS_REGISTRATION","REGION_POPULATION_RELATIVE",
            "AMT_CREDIT" , "AMT_INCOME_TOTAL","NAME_CONTRACT_TYPE", "OWN_CAR_AGE", "ORGANIZATION_TYPE"
             ]

target = train["TARGET"].values
train = train[use_features]
train["TARGET"] = target
test = test[use_features]

# EXT_SOURCE_2の欠損値を平均値で補完
train["EXT_SOURCE_2"].fillna(train["EXT_SOURCE_2"].mean(), inplace=True)
test["EXT_SOURCE_2"].fillna(train["EXT_SOURCE_2"].mean(), inplace=True)
train["DAYS_LAST_PHONE_CHANGE"].fillna(train["DAYS_LAST_PHONE_CHANGE"].mean(), inplace=True)
train["AMT_GOODS_PRICE"].fillna(train["AMT_GOODS_PRICE"].mean(), inplace=True)
test["AMT_GOODS_PRICE"].fillna(train["AMT_GOODS_PRICE"].mean(), inplace=True)
train.isnull().sum()

columns = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
# train データフレームの処理: 欠損値を無視して平均値を計算
train_ext_source_mean = np.nanmean(train[columns], axis=1)
train_ext_source_mean = pd.DataFrame(train_ext_source_mean, columns=["EXT_SOURCE_MEAN"])
train = pd.concat([train, train_ext_source_mean], axis=1)

test_ext_source_mean = np.nanmean(test[columns], axis=1)
test_ext_source_mean = pd.DataFrame(test_ext_source_mean, columns=["EXT_SOURCE_MEAN"])
test = pd.concat([test, test_ext_source_mean], axis=1)

train.drop('EXT_SOURCE_1', axis=1, inplace=True)
test.drop('EXT_SOURCE_1', axis=1, inplace=True)
train.drop('EXT_SOURCE_2', axis=1, inplace=True)
test.drop('EXT_SOURCE_2', axis=1, inplace=True)
train.drop('EXT_SOURCE_3', axis=1, inplace=True)
test.drop('EXT_SOURCE_3', axis=1, inplace=True)

train_credit_goods_price_ratio = train['AMT_CREDIT'] / train['AMT_GOODS_PRICE']
train_credit_goods_price_ratio = pd.DataFrame(train_credit_goods_price_ratio, columns=["CREDIT_GOODS_PRICE_RATIO"])
train = pd.concat([train, train_credit_goods_price_ratio], axis=1)
test_credit_goods_price_ratio = test['AMT_CREDIT'] / test['AMT_GOODS_PRICE']
test_credit_goods_price_ratio = pd.DataFrame(test_credit_goods_price_ratio, columns=["CREDIT_GOODS_PRICE_RATIO"])
test = pd.concat([test, test_credit_goods_price_ratio], axis=1)

train_credit_annual_ratio = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
train_credit_annual_ratio = pd.DataFrame(train_credit_annual_ratio, columns=["CREDIT_ANNUAL_RATIO"])
train = pd.concat([train, train_credit_annual_ratio], axis=1)
test_credit_annual_ratio = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']
test_credit_annual_ratio = pd.DataFrame(test_credit_annual_ratio, columns=["CREDIT_ANNUAL_RATIO"])
test = pd.concat([test, test_credit_annual_ratio], axis=1)

train_credit_down_payment = train['AMT_GOODS_PRICE'] - train['AMT_CREDIT']
train_credit_down_payment = pd.DataFrame(train_credit_down_payment, columns=["CREDIT_DOWN_PAYMENT"])
train = pd.concat([train, train_credit_down_payment], axis=1)
test_credit_down_payment = test['AMT_GOODS_PRICE'] - test['AMT_CREDIT']
test_credit_down_payment = pd.DataFrame(test_credit_down_payment, columns=["CREDIT_DOWN_PAYMENT"])
test = pd.concat([test, test_credit_down_payment], axis=1)

train.drop('AMT_CREDIT', axis=1, inplace=True)
test.drop('AMT_CREDIT', axis=1, inplace=True)
train.drop('AMT_GOODS_PRICE', axis=1, inplace=True)
test.drop('AMT_GOODS_PRICE', axis=1, inplace=True)

# NAME_CONTRACT_TYPEの数値化（Label Encoding）
train["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}, inplace=True)
test["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}, inplace=True)

# ORGANIZATION_TYPEの数値化（Count Encoding）
organization_ce = train["ORGANIZATION_TYPE"].value_counts()
train["ORGANIZATION_TYPE"] = train["ORGANIZATION_TYPE"].map(organization_ce)
test["ORGANIZATION_TYPE"] = test["ORGANIZATION_TYPE"].map(organization_ce)

# OWN_CAR_AGEの60以上の値（外れ値）を欠損値扱いする
train.loc[train["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan
test.loc[test["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan
# OWN_CAR_AGEをグループ分け
train["OWN_CAR_AGE"] = train["OWN_CAR_AGE"] // 10
test["OWN_CAR_AGE"] = test["OWN_CAR_AGE"] // 10

train["OWN_CAR_AGE"].unique()
# OWN_CAR_AGEをOne Hot Encoding
train_car_age_ohe = pd.get_dummies(train["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_")
test_car_age_ohe = pd.get_dummies(test["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_")

train = pd.concat([train, train_car_age_ohe], axis=1)
test = pd.concat([test, test_car_age_ohe], axis=1)

train.drop('OWN_CAR_AGE', axis=1, inplace=True)
test.drop('OWN_CAR_AGE', axis=1, inplace=True)

# ライブラリの読み込み
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
# 目的変数と説明変数に分割

X = train.drop("TARGET", axis=1).values
y = train["TARGET"].values
X_test = test.values
# 標準化
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
X_test_std = sc.transform(X_test)
# 訓練データと評価データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=42)





from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, DMatrix
from xgboost.callback import EarlyStopping
from lightgbm import log_evaluation



def objective_lgbm(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 0
    }

    model = LGBMClassifier(**params)
    
    # callbacksで早期終了を指定
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=callbacks) 
    preds = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, preds)
    return roc_auc

study_lgbm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study_lgbm.optimize(objective_lgbm, n_trials=1000, n_jobs=-1)

print('Best trial:', study_lgbm.best_trial.params)
print('Best score:', study_lgbm.best_value)

best_params_lgbm = study_lgbm.best_params
best_lgbm = LGBMClassifier(**best_params_lgbm)
best_lgbm.fit(X_train, y_train)

lgbm_train_pred = best_lgbm.predict_proba(X_train)[:, 1]
lgbm_valid_pred = best_lgbm.predict_proba(X_valid)[:, 1]
print(f"Train Score: {roc_auc_score(y_train, lgbm_train_pred)}")
print(f"Valid Score: {roc_auc_score(y_valid, lgbm_valid_pred)}")





def objective(trial):
    
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'random_state': 0
    }

    # DMatrixを作成
    dtrain = DMatrix(X_train, label=y_train)
    dvalid = DMatrix(X_valid, label=y_valid)

    # EarlyStopping コールバックを作成
    early_stop = EarlyStopping(rounds=100, metric_name="auc", maximize=False)

    # xgboost.training.train を使用して学習
    bst = xgb.train(param, dtrain, evals=[(dvalid, 'eval')], callbacks=[early_stop], verbose_eval=False)

    # 最良のモデルで予測
    preds = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1)) 
    roc_auc = roc_auc_score(y_valid, preds)
    return roc_auc


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=1000, n_jobs=-1)  # 試行回数

print('Best trial:', study.best_trial.params)
print('Best score:', study.best_value)

# 最適なハイパーパラメータでモデルを再学習
best_params = study.best_params
best_xgb = XGBClassifier(**best_params)
best_xgb.fit(X_train, y_train)


xgb_train_pred = best_xgb.predict_proba(X_train)[:, 1]
xgb_valid_pred = best_xgb.predict_proba(X_valid)[:, 1]
print(f"Train Score: {roc_auc_score(y_train, xgb_train_pred)}")
print(f"Valid Score: {roc_auc_score(y_valid, xgb_valid_pred)}")





# アンサンブル (単純な平均)
pred_lgbm = best_lgbm.predict_proba(X_test)[:, 1]
pred_xgb = best_xgb.predict_proba(X_test)[:, 1]
pred_ensemble = (pred_lgbm + pred_xgb) / 2

# 最良モデルの選択と予測
models = {'LightGBM': best_lgbm, 'XGBoost': best_xgb, 'Ensemble': None}
best_model_name = max(models, key=lambda model_name: roc_auc_score(y_valid, models[model_name].predict_proba(X_valid)[:, 1]))

# 最良モデルで予測
best_model = models[best_model_name]
pred = best_model.predict_proba(X_test)[:, 1]

print(f'Best Model: {best_model_name}')

# 提出ファイルの作成
sample_sub['TARGET'] = pred
sample_sub
sample_sub.to_csv('submission.csv', index=False)
