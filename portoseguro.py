import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
train_data = pd.read_csv('train.csv',index_col='id')
train_data.head()

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from numba import jit

# Compute gini
# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def normalized_gini(solution, submission):
    normalized_gini = eval_gini(solution, submission)/eval_gini(solution, solution)
    return normalized_gini

# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

#def gini_xgb(preds, dtrain):
#    labels = dtrain.get_label()
#    gini_score = -eval_gini(labels, preds)
#    return [('gini', gini_score)]

gini_scorer = make_scorer(normalized_gini)

# Regularized Greedy Forest
#from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python


train = pd.read_csv(r'F:\Juhon\Koodi\python\kaggle-porto_seguro\train.csv')
test = pd.read_csv(r'F:\Juhon\Koodi\python\kaggle-porto_seguro\test.csv')

# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

def preproc(X_train):
    # Adding new features and deleting features with low importance
    multreg = X_train['ps_reg_01'] * X_train['ps_reg_03'] * X_train['ps_reg_02']
    ps_car_reg = X_train['ps_car_13'] * X_train['ps_reg_03'] * X_train['ps_car_13']
    X_train = X_train.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',
                            'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                            'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_car_10_cat', 'ps_ind_10_bin',
                            'ps_ind_13_bin', 'ps_ind_12_bin'], axis=1)
    X_train['mult'] = multreg
    X_train['ps_car'] = ps_car_reg
    X_train['ps_ind'] = X_train['ps_ind_03'] * X_train['ps_ind_15']
    return X_train

train = preproc(train)
test = preproc(test)

print(train.values.shape, test.values.shape)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)

print(train.values.shape, test.values.shape)

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
    
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=10, n_jobs=1, scoring=gini_scorer)
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['random_state'] = 99


# RandomForest params
rf_params = {}
rf_params['n_estimators'] = 200
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30


# ExtraTrees params
et_params = {}
et_params['n_estimators'] = 155
et_params['max_features'] = 0.3
et_params['max_depth'] = 6
et_params['min_samples_split'] = 40
et_params['min_samples_leaf'] = 18


# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 490
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9  
xgb_params['min_child_weight'] = 10

xgb_params2 = {}
xgb_params2['objective'] = 'binary:logistic'
xgb_params2['learning_rate'] = 0.07
xgb_params2['n_estimators'] = 400
xgb_params2['max_depth'] = 4
xgb_params2['subsample'] = 0.8
xgb_params2['colsample_bytree'] = 0.8  
xgb_params2['min_child_weight'] = 6
xgb_params2['scale_pos_weight'] = 1.6
xgb_params2['gamma'] = 10
xgb_params2['reg_alpha'] = 8
xgb_params2['reg_lambda'] = 1.3

xgb_params3 = {}
xgb_params3['objective'] = 'binary:logistic'
xgb_params3['learning_rate'] = 0.1
xgb_params3['n_estimators'] = 400
xgb_params3['max_depth'] = 4
xgb_params3['subsample'] = 0.9
xgb_params3['colsample_bytree'] = 0.9  
xgb_params3['min_child_weight'] = 10

# CatBoost params
cat_params = {}
cat_params['iterations'] = 900
cat_params['depth'] = 8
cat_params['rsm'] = 0.95
cat_params['learning_rate'] = 0.03
cat_params['l2_leaf_reg'] = 3.5  
cat_params['border_count'] = 8
cat_params['gradient_iterations'] = 4


# Regularized Greedy Forest params
rgf_params = {}
rgf_params['max_leaf'] = 2000
rgf_params['learning_rate'] = 0.5
rgf_params['algorithm'] = "RGF_Sib"
rgf_params['test_interval'] = 100
rgf_params['min_samples_leaf'] = 3 
rgf_params['reg_depth'] = 1.0
rgf_params['l2'] = 0.5  
rgf_params['sl2'] = 0.005



lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

rf_model = RandomForestClassifier(**rf_params)

et_model = ExtraTreesClassifier(**et_params)
        
xgb_model = XGBClassifier(**xgb_params)

xgb_model2 = XGBClassifier(**xgb_params2)

xgb_model3 = XGBClassifier(**xgb_params3)

cat_model = CatBoostClassifier(**cat_params)

#rgf_model = RGFClassifier(**rgf_params)

gb_model = GradientBoostingClassifier(max_depth=5)

ada_model = AdaBoostClassifier()

log_model = LogisticRegression()


        
stack = Ensemble(n_splits=10,
        stacker = log_model,
        base_models = (lgb_model,
                       lgb_model2,
                       lgb_model3,
#                       rf_model,
#                       et_model,
                       xgb_model,
                       xgb_model2,
                       xgb_model3,
#                       cat_model,
#                       rgf_model,
#                       gb_model,
#                       ada_model,
#                       log_model
                       )
        )        
        
y_pred = stack.fit_predict(train, target_train, test)



sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv(r'subs\stacked_4.csv', index=False)



