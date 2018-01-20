#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
from random import seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.ticker as ticker

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from statsmodels.graphics import mosaicplot
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import fancyimpute

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def pre_process(df, survived_names):
    df['Deck'] = df['Cabin'].dropna().apply(lambda x: x[0])
#    df = df.join(pd.get_dummies(df['Deck']))
#    df.Sex = df['Sex'].map({'female': 0, 'male':1}).astype(int)
#    for col in ['Fare',]:
#        df = fillnas(df, col)
#    embark_dummies  = pd.get_dummies(df['Embarked'])
#    embark_dummies.drop(['S'], axis=1, inplace=True)
#    df = df.join(embark_dummies)
#    df.drop('Embarked', axis=1, inplace=True)
#    df = fillnas(df, 'Age')
    df['Namelen'] = df['Name'].map(lambda x: len(x))

    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = ['Mr','Master','Miss','Mrs']
    df['Title'] = df['Title'].apply(lambda x: 'Rare' if x not in titles else x)
#    df = df.join(pd.get_dummies(df['Title']))
    df['Fsize'] = df.SibSp + df.Parch
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('SibSp', axis=1, inplace=True)
    df.drop('Parch', axis=1, inplace=True)
    df['FsizeD'] = df['Fsize'].apply(discr_Fsize)        
    df.drop('Fsize', axis=1, inplace=True)
    
    df['NamelenD'] = df['Namelen'].apply(discr_Namelen)
    df.drop('Namelen', axis=1, inplace=True)
    
#    df['Mother'] = (df['Title'] == 'Mrs') & (df['Parch'] > 0)
#    df['Mother'] = df['Mother'].astype(int)
    
    df['FamilySurvived'] = 0
    df['FamilyName'] = [re.split(', ', name)[0] for name in df['Name']]
    last_digits = [ticket[-3:] for ticket in df['Ticket']]
    df['FamilyName'] = df['FamilyName'] + last_digits
    df.loc[df['FamilyName'].isin(survived_names), 'FamilySurvived'] = 1
    df.drop('FamilyName', axis=1, inplace=True)
    
    df.drop('Ticket', axis=1, inplace=True)

#    df.drop('Embarked', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df = pd.get_dummies(df)
#    df = dectree_impute(df) # only fare
    df = fancy_impute(df) #, plotcol='Age') # imputes all missing -> plot to check
    # discretized only after imputation
    df['AgeD'] = df['Age'].apply(discr_Age)
    df['FareD'] = df['Fare'].apply(discr_Fare)
    df.drop('Age', axis=1, inplace=True)
    df.drop('Fare', axis=1, inplace=True)
    
    df = pd.get_dummies(df)
#    df = normalize_features(df)
    return(df)

def discr_Fsize(size):
    if size < 1:
        size = 'single'
    elif size < 4:
        size = 'small'
    else:
        size = 'large'
    return(size)
    
def discr_Age(age):
    if age < 14:
        age = 'child'
    elif age < 35:
        age = 'young'
    else:
        age = 'old'
    return(age)

def discr_Fare(fare):
    if fare < 30:
        fare = 'low'
    else:
        fare = 'high'
    return(fare)

def discr_Namelen(namelen):
    if namelen < 13:
        namelen = 'short'
    elif namelen < 29:
        namelen = 'mid'
    else:
        namelen = 'long'
    return(namelen)

def fillnas(df, col):
    if df[col].isnull().sum() == 0:
        return(df)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    nan_no = df[col].isnull().sum()
    print(col + ': ' + str(round(nan_no/len(df[col])*100, 2)) + ' % NaN')
#    print(df[col].value_counts())
    if df[col].dtype in numerics:
        rand = np.random.randint(df[col].mean() - df[col].std(), df[col].mean()  + df[col].std(), size = nan_no)
        df[col][np.isnan(df[col])] = rand
        print('Filled by random around mean\n')
    elif col == 'Embarked':
        df[col].fillna('C') # due to matching 'Fare'
    else:
        most_common = df[col].value_counts().idxmax()
        df[col].fillna(most_common, inplace=True)
        print('Filled  by ' + str(most_common) + '\n')
    return(df)

def dectree_impute(df):
    features = [f for f in df.columns if f not in ['Fare', 'Age']]
    print(features)
    all_fares = df.loc[pd.notnull(df['Fare'])]
    missing_fares = df.loc[pd.isnull(df['Fare'])]
    X = all_fares[features]
    y = all_fares['Fare']
    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(X,y)
    dectree_imputed = clf.predict(missing_fares[features])
    df.loc[pd.isnull(df['Fare']), 'Fare'] = dectree_imputed
#    print(dectree_imputed)
#    print(df.Fare.mean())
    return(df)

def fancy_impute(df, plotcol=False):
    imputed = fancyimpute.MICE(n_imputations=6, impute_type='col').complete(np.array(df))
    imputed = pd.DataFrame(imputed, columns = list(df))
    if plotcol:
        plt.figure()
        sns.kdeplot(imputed[plotcol], label = 'imputed')
        ax = sns.kdeplot(df[plotcol].dropna(), label = 'original')
        ax.set(xlabel=plotcol, ylabel='Density');
        plt.show()
    df = imputed
    return(df)    
    
def normalize_features(df):
    df = df.astype(float)
    df = df/df.max()
    
    # better normalization?
    # ok for all methods?
    return(df)
    
def preview_groupby_mean(df, Y):
    # groupby var and get mean for quick corr check
    plt.figure(figsize=(15,10))
    for col in df.columns:
        ind = df.columns.get_loc(col)
        plt.subplot(3, ceil(len(df.columns)/3), ind+1)
        if col != Y:
    #        print(col)
            perc = df[[col, "Survived"]].groupby(col,as_index=False).mean()
    #        print(perc)
            plt.plot(perc[col], perc[Y], 'o')
            plt.title(col)
    
def pairplot_numeric(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    for col in df.columns:
        if df[col].isnull().any():
            df = df.drop(col, axis=1)
#    df.dropna() # drops all rows with any NaN
    sns.pairplot(df)
    return()
    
def facetplot(df, col):
    facet = sns.FacetGrid(df, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,col,shade= True)
    facet.set(xlim=(0, titanic_df[col].max()))
    facet.add_legend()

def get_sets(df, Y, test_df, CV_fract = 0.25):
    if df.isnull().sum().sum() == 0 and df.dtypes.all() == float \
    and test_df.isnull().sum().sum() == 0 and test_df.dtypes.all() == float:
        len_train = int(len(df)*(1-CV_fract))
        X_train = df.drop(range(len_train,len(df))).drop(Y ,axis=1)
        X_CV = df.drop(range(len_train)).drop(Y,axis=1)
        Y_train = df.drop(range(len_train,len(df)))[Y]
        Y_CV = df.drop(range(len_train))[Y]
        X_test  = test_df
    else:
        print(df.describe())
        print(df.info())
        print('Data not clean')
    return(X_train, Y_train, X_CV, Y_CV, X_test)
    
def classify_labeled_meta(X_train, Y_train, X_CV, Y_CV, X_test):
    # compare sklearn methods based on CV resultd
    results = {}
#    models = 
    
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    results['LogReg'] = [logreg.score(X_train, Y_train), logreg.score(X_CV, Y_CV)]
    Y_pred = logreg.predict(X_test)
    best_CV_score = logreg.score(X_CV, Y_CV)
    
#    # Ridge
#    logreg_ridge = LogisticRegressionCV(Cs=100, cv=10, penalty='l2')#, random_state=0)
#    logreg_ridge.fit(X_train, Y_train)
#    results['Ridge'] = [logreg_ridge.score(X_train, Y_train), logreg_ridge.score(X_CV, Y_CV)]
#    if results['Ridge'][1] > best_CV_score:
#        best_CV_score = results['Ridge'][1]
#        Y_pred = logreg_ridge.predict(X_test)
#    
#    # Lasso
#    logreg_lasso = LogisticRegressionCV(Cs=100, cv=10, penalty='l1', solver='liblinear')#, random_state=0)
#    logreg_lasso.fit(X_train, Y_train)
#    results['Lasso'] = [logreg_lasso.score(X_train, Y_train), logreg_lasso.score(X_CV, Y_CV)]
#    if results['Lasso'][1] > best_CV_score:
#        best_CV_score = results['Lasso'][1]
#        Y_pred = logreg_lasso.predict(X_test)
    
    # Support Vector Machines
    # svc = LinearSVC()
    svc = SVC(kernel='linear', tol=1e-5) #class_weight={'Survived':1,'Pclass':1,'Sex':1,'Age':1,'SibSp':1,'Parch':1,'Fare':1,'C':1,'Q':1,'Namelen':1})
    svc.fit(X_train, Y_train)
    results['SVC'] = [svc.score(X_train, Y_train), svc.score(X_CV, Y_CV)]
    if results['SVC'][1] > best_CV_score:
        best_CV_score = results['SVC'][1]
    Y_pred = svc.predict(X_test)
    
#    # Random Forests (0.78947 with n_estimators=5)
#    random_forest = RandomForestClassifier(random_state=123,n_estimators=3)
#    random_forest.fit(X_train, Y_train)
#    results['RandomForest'] = [random_forest.score(X_train, Y_train), random_forest.score(X_CV, Y_CV)]
#    if results['RandomForest'][1] > best_CV_score:
#        best_CV_score = results['RandomForest'][1]
#        Y_pred = random_forest.predict(X_test)
        
#    # Knn (0.78469 submission with default + n_neighbors=3)
#    knn = KNeighborsClassifier(n_neighbors = 3)
#    knn.fit(X_train, Y_train)
#    results['knn'] = [knn.score(X_train, Y_train), knn.score(X_CV, Y_CV)]
#    if results['knn'][1] > best_CV_score:
#        best_CV_score = results['knn'][1]
#        Y_pred = knn.predict(X_test)
    
#    # Gaussian Naive Bayes
#    GNB = GaussianNB()
#    GNB.fit(X_train, Y_train)
#    results['GNB'] = [GNB.score(X_train, Y_train), GNB.score(X_CV, Y_CV)]
#    if results['GNB'][1] > best_CV_score:
#        best_CV_score = results['GNB'][1]
#        Y_pred = GNB.predict(X_test)

    return(results, Y_pred, logreg.coef_[0])

###########################################################################

# Survived families
df = pd.read_csv('train.csv', dtype={"Age": np.float64}, )
df['FamilyName'] = [re.split(', ', name)[0] for name in df['Name']]
family_ticket = df.loc[df['Parch'] + df['SibSp'] > 0, ['FamilyName', 'Ticket']]
#family_ticket = family_ticket.sort_values('FamilyName')
#family_ticket.head()
last_digits = [ticket[-3:] for ticket in df['Ticket']]
df['FamilyName'] = df['FamilyName'] + last_digits
families = df.loc[df['Parch'] + df['SibSp'] > 0]
family_survived = families.groupby(['FamilyName'])['Survived'].sum()
family_survived = family_survived[family_survived > 0]
survived_names = family_survived.index.values

Xname, Yname = 'PassengerId', 'Survived'
Y_pred_all = {}
for sex in ['male', 'female']:
    print('\n' + sex + ':\n')
    train = pd.read_csv('train.csv', dtype={"Age": np.float64}, )
    #train.reindex(np.random.permutation(train.index))
    test = pd.read_csv('test.csv', dtype={"Age": np.float64}, )
    train = train[train['Sex']==sex]
    test = test[test['Sex']==sex]
    len_train = len(train)
    Ycol = train[Yname]
    Ycol.reset_index(drop=True, inplace=True)    
    full = train.drop(Yname, axis=1).append(test, ignore_index=True).drop(Xname, axis=1)
    full = full.drop('Sex', axis=1)
    full = pre_process(full, survived_names)
#    full = pre_process(full)
    train = full[:len_train].join(Ycol)
    test = full[len_train:]
    
    seed(123)
#    X_train, Y_train, X_CV, Y_CV, X_test = get_sets(train, Yname, test)
#    results, Y_pred, logreg_coeffs = classify_labeled_meta(X_train, Y_train, X_CV, Y_CV, test)
    
#    coeff_df = pd.DataFrame(train.columns.delete(0))
#    coeff_df.columns = ['Feature']
#    coeff_df["Coeff_estimate"] = pd.Series(logreg_coeffs).abs()
#    print(coeff_df.sort_values('Coeff_estimate'))
#    
#    for key in results:
#        print(key, results[key])

    

    X_train, X_test, y_train, y_test = train_test_split(
    train.drop(Yname,axis=1), Ycol, test_size=0.2, random_state=0)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.05, 0.01],
                         'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
#    scores = ['precision', 'recall']
#    
#    for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3)
#                       scoring='%s_macro' % score)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


    Y_pred_all[sex] = clf.predict(test)
#    Y_pred_all[sex] = Y_pred

test = pd.read_csv('test.csv', dtype={"Age": np.float64}, )
X_submit = test[Xname]

sub_male = pd.DataFrame({
        "PassengerId": test[test['Sex']=='male']['PassengerId'],
        "Survived": Y_pred_all['male'].astype(int)
    })

sub_female = pd.DataFrame({
        "PassengerId": test[test['Sex']=='female']['PassengerId'],
        "Survived": Y_pred_all['female'].astype(int)
    })

submission = sub_male.combine_first(sub_female).astype(int)
submission.to_csv('titanic.csv', index=False)

#submission = pd.DataFrame({
#        "PassengerId": X_submit,
#        "Survived": Y_pred.astype(int)
#    })
#submission.to_csv('titanic.csv', index=False)

# todo:
# use CV from skl
# read up on sklearn methods
# map Pclass ABC
# map optimal parameters, e.g. RF error vs trees



## preview
#print(train_df.head())
#pd.crosstab(train.Survived,train.Sex)
#pd.crosstab(train.Survived,train.Pclass, normalize=True)
#pd.crosstab(pd.isnull(train['Age']), train.Pclass, normalize=True)
#preview_groupby_mean(train_df, Yname)
#pairplot_numeric(train_df)
#facetplot(train_df, 'Age')
#facetplot(train_df, 'Fare')
#facetplot(train_df, 'Pclass')
#facetplot(train_df, 'Namelen')
#g = sns.factorplot(x='Age', y='PassengerId', hue='Survived', col='Sex', kind='strip', data=train);
#ax = plt.gca()
#ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))
#plt.show()