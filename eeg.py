# -*- coding: utf-8 -*-
import os
import time
from math import log
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from threading import Thread
#from Queue import Queue
import multiprocessing

from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
#from sklearn.cross_validation import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from itertools import combinations
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.linear_model import LogisticRegression, Ridge
#from sklearn.datasets import dump_svmlight_file

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#def ieegMatToPandasDF(path):
#    mat = loadmat(path)
#    names = mat['dataStruct'].dtype.names
#    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
#    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])   
#
#def ieegMatToArray(path):
#    mat = loadmat(path)
#    names = mat['dataStruct'].dtype.names
#    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
#    return ndata['data']  

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return(ndata)

def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x

#def ieegSingleMetaData(path):
#    mat_data = scipy.io.loadmat(path)
#    data = mat_data['dataStruct']
#    for i in [data, data[0], data[0][0], data[0][0][0], data[0][0][0][0]]:
#        print((i.shape, i.size))

# EEG clips labeled "Preictal" (k=1) for pre-seizure data segments, 
# or "Interictal" (k-0) for non-seizure data segments.
# I_J_K.mat - the Jth training data segment corresponding to the Kth 
# class (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three patients).

#def bin_power(X,Band,Fs):
##	Returns
##	Power:list,spectral power in each frequency bin.
##	Power:ratio,list,spectral power in each frequency bin normalized by total power in ALL frequency bins.
#	C = abs(np.fft.fft(X))
#	Power =np.zeros(len(Band)-1);
#	for Freq_Index in range(len(Band)-1):
#		Freq = float(Band[Freq_Index])
#		Next_Freq = int(Band[Freq_Index+1])
#		Power[Freq_Index] = sum(C[int(np.floor(Freq/Fs*len(X))):int(np.floor(Next_Freq/Fs*len(X)))])
#	Power_Ratio = Power/sum(Power)
#	return(Power, Power_Ratio)

#def get_features(x):
#    Fs = 400 # sampling rate
#    bins = [0.5,4,7,13,26,100] # from https://www.epilepsysociety.org.uk/closer-look-eeg
#    # also "spike" <80ms, "sharp" 80..200ms, "polyspike", "spike wave"
#    P, ratios = bin_power(x[:,0],bins, Fs)
#    return(ratios)

def get_features(file):
    f = mat_to_data(file)
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
#    print((nt, nc))
    subsampLen = int(np.floor(fs * 60))
    numSamps = int(np.floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,(numSamps+1)*subsampLen,subsampLen)
    feat = [] # Feature Vector
    for i in range(1, numSamps+1):
#        print('processing file {} epoch {}'.format(file,i))
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]
    
        # compute Shannon's entropy, spectral edge and correlation matrix
        # segments corresponding to frequency bands
        lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        lseg = np.round(nt/fs*lvl).astype('int')
        D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
        D[0,:]=0                                # set the DC component to zero
        D /= D.sum()                      # Normalize each channel               
        
        dspect = np.zeros((len(lvl)-1,nc))
        for j in range(len(dspect)):
            dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)
    
        # Find the shannon's entropy
        spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
    
        # Find the spectral edge frequency
        sfreq = fs
        tfreq = 40
        ppow = 0.5
    
        topfreq = int(round(nt/sfreq*tfreq))+1
        A = np.cumsum(D[:topfreq,:])
        B = A - (A.max()*ppow)
        spedge = np.min(np.abs(B))
        spedge = (spedge - 1)/(topfreq-1)*tfreq
    
        # Calculate correlation matrix and its eigenvalues (b/w channels)
        data = pd.DataFrame(data=epoch)
        type_corr = 'pearson'
        lxchannels = corr(data, type_corr)
        
        # Calculate correlation matrix and its eigenvalues (b/w freq)
        data = pd.DataFrame(data=dspect)
        lxfreqbands = corr(data, type_corr)
        
        # Spectral entropy for dyadic bands
        # Find number of dyadic levels
        ldat = int(np.floor(nt/2.0))
        no_levels = int(np.floor(log(ldat,2.0)))
        seg = np.floor(ldat/pow(2.0, no_levels-1))
    
        # Find the power spectrum at each dyadic level
        dspect = np.zeros((no_levels,nc))
        for j in range(no_levels-1,-1,-1):
            dspect[j,:] = 2*np.sum(D[int(np.floor(ldat/2.0))+1:ldat,:], axis=0)
            ldat = int(np.floor(ldat/2.0))
    
        # Find the Shannon's entropy
        spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
    
        # Find correlation between channels
        data = pd.DataFrame(data=dspect)
        lxchannelsDyd = corr(data, type_corr)
        
        # Fractal dimensions
        no_channels = nc
        #fd = np.zeros((2,no_channels))
        #for j in range(no_channels):
        #    fd[0,j] = pyeeg.pfd(epoch[:,j])
        #    fd[1,j] = pyeeg.hfd(epoch[:,j],3)
        #    fd[2,j] = pyeeg.hurst(epoch[:,j])
    
        #[mobility[j], complexity[j]] = pyeeg.hjorth(epoch[:,j)
        # Hjorth parameters
        # Activity
        activity = np.var(epoch, axis=0)
        #print('Activity shape: {}'.format(activity.shape))
        # Mobility
        mobility = np.divide(
                            np.std(np.diff(epoch, axis=0)), 
                            np.std(epoch, axis=0))
        #print('Mobility shape: {}'.format(mobility.shape))
        # Complexity
        complexity = np.divide(np.divide(
                                        # std of second derivative for each channel
                                        np.std(np.diff(np.diff(epoch, axis=0), axis=0), axis=0),
                                        # std of second derivative for each channel
                                        np.std(np.diff(epoch, axis=0), axis=0))
                               , mobility)
        #print('Complexity shape: {}'.format(complexity.shape))
        # Statistical properties
        # Skewness
        sk = skew(epoch)
    
        # Kurtosis
        kurt = kurtosis(epoch)
    
        # compile all the features
        feat = np.concatenate((feat,
                               spentropy.ravel(),
                               spedge.ravel(),
                               lxchannels.ravel(),
                               lxfreqbands.ravel(),
                               spentropyDyd.ravel(),
                               lxchannelsDyd.ravel(),
                               #fd.ravel(),
                               activity.ravel(),
                               mobility.ravel(),
                               complexity.ravel(),
                               sk.ravel(),
                               kurt.ravel()
                                ))
        return(feat)
    
def get_train_data(train_path, number_of_features):
    safe_files = pd.read_csv('train_and_test_data_labels_safe.csv')
    safe_files = safe_files[safe_files.safe==1].image.values
    files = [f for f in os.listdir(train_path) if f in safe_files]
    len_files = len(files)
    print(str(len(files)) + '/' + str(len(os.listdir(train_path))) + ' train files safe')
    feature_matrix, y = np.zeros((len_files,number_of_features)), np.zeros((len_files,))
    for file in files:
        path = train_path + file
        features = get_features(path)
        ind = files.index(file)
        if ind % 100 == 0:
            print(round(ind/len_files*100))
        feature_matrix[ind,:] = features
        y[ind] = int(file[-5])
    nanbool = np.isnan(feature_matrix).any(axis=1)
    X_train = preprocessing.scale(feature_matrix[~nanbool,:], axis=1)
    y = y[~nanbool]
    print('Train: ' + str(sum(nanbool)) + ' NaN-rows removed')
    return(X_train, y)
    
def get_test_data(test_path, number_of_features):
    files = [f for f in os.listdir(test_path)]
    len_files = len(files)
    feature_matrix = np.zeros((len_files,number_of_features))
    for file in files:
        path = test_path + file
        features = get_features(path)
        ind = files.index(file)
        if ind % 100 == 0:
            print(round(ind/len_files*100))
        feature_matrix[ind,:] = features
    nanbool = np.isnan(feature_matrix).any(axis=1)
    X_test = preprocessing.scale(feature_matrix[~nanbool,:], axis=1)
    print('Test: ' + str(sum(nanbool)) + ' NaN-rows removed')
    return(X_test, files, nanbool)

def train_LinearSVC(train, y, test):
    X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2, random_state=0)
    
    # Set the parameters by cross-validation
#    tuned_parameters = [{'n_estimators': [1, 3, 5, 10, 15, 20]},
#                        ]
                        
    tuned_parameters = [{'C': [1e4] #,10, 100, 1000],
                        },
                        ]
    
#    scores = ['precision']#, 'recall']
#    
#    for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()

    clf = GridSearchCV(LinearSVC(), tuned_parameters,)
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

    Y_pred = clf.predict(test)
    return(Y_pred)
    
def train_SVC(train, y, test):
    X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2, random_state=0)
    
    # Set the parameters by cross-validation
#    tuned_parameters = [{'n_estimators': [1, 3, 5, 10, 15, 20]},
#                        ]
                        
    tuned_parameters = [{'kernel': ['rbf'],
                         'gamma': [1e-3,1/160,1e-2,1e-1],
#                         'gamma': [0.15],
                         'C': np.arange(1,20,1),
#                         'C': [1],
                        },
                        ]
    
#    scores = ['precision', 'recall']
#    
#    for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(class_weight='balanced',
                           probability=True,
                           ),
                       tuned_parameters,
                       n_jobs=-1,
#                       scoring='recall',
#                       scoring='%s_macro' % score,
                       )

#    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=3)
    
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
    
    Y_pred = clf.predict(test)
    Y_probs = clf.predict_proba(test)
    return(Y_pred, Y_probs)
    
def train_RF(train, y, test):
    X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2, random_state=0)
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [3],
                         'max_features': [3,5,10,15,20],
                        },
                        ]
                        
#    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 5e-3, 1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
#    scores = ['precision', 'recall']
#    
#    for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
    print()

#    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3)
#                       scoring='%s_macro' % score)

    clf = GridSearchCV(RandomForestClassifier(class_weight='balanced_subsample'),#'balanced'),
                       tuned_parameters,
                       n_jobs=-1,
#                       scoring='recall',
                       )
    
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
    
    Y_pred = clf.predict(test)
    return(Y_pred)
    
def train_KNN(train, y, test):
    X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.4, random_state=0)
    
    # Set the parameters by cross-validation
#    tuned_parameters = [{'n_estimators': [1, 3, 5, 10, 15, 20]},
#                        ]
                        
    tuned_parameters = [{'n_neighbors': [5,10,20,100]
                        },
                        ]
    
#    scores = ['precision']#, 'recall']
#    
#    for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()

    clf = GridSearchCV(KNeighborsClassifier(weights='distance'),
                       tuned_parameters,
                       n_jobs=-1,
                       scoring='recall',
                        )

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

    Y_pred = clf.predict(test)
    return(Y_pred)
    

## get features and save to file
#train_paths = ['/media/juho/Data/Juhon/data/train/train_1/',
#               '/media/juho/Data/Juhon/data/train/train_2/',
#               '/media/juho/Data/Juhon/data/train/train_3/',
##               '/media/juho/Data/Juhon/data/eegtest/train/',
#               ]
#test_paths = ['/media/juho/Data/Juhon/data/test/test_1_new/',
#              '/media/juho/Data/Juhon/data/test/test_2_new/',
#              '/media/juho/Data/Juhon/data/test/test_3_new/',
##              '/media/juho/Data/Juhon/data/eegtest/test/',
#              ]
#testfile = '/media/juho/Data/Juhon/data/train/train_1/1_10_0.mat'
#testfile_features = get_features(testfile)
#number_of_features = len(testfile_features)
#print(str(number_of_features) + ' features')
#
#all_features_train = []
#all_y = []
#all_features_test = []
#all_files = []
#all_nanbools = []
#for trp,tep in zip(train_paths, test_paths):
#    print(trp,tep)
#        
#    t = time.time()
#    
#    #q = Queue()
#    #Thread(target=get_train_data, args=[train_path, number_of_features]).start()
#    #Thread(target=get_test_data, args=[test_path, number_of_features]).start()
#    #Y_pred = train_model(X_train, y, X_test, t)
#    
##    pool = multiprocessing.Pool()
##    result_train = pool.apply_async(get_train_data, (train_path, number_of_features))
##    result_test = pool.apply(get_test_data, (test_path, number_of_features))
#    
#    #X_train, y = result_train.get()
#    #X_test = result_test#.get()
#
#    train, y = get_train_data(trp, number_of_features)
#    test, files, nanbool =  get_test_data(tep, number_of_features)
#    print('Get features: ' + str(time.time() - t))
#    all_features_train.append(train)
#    all_y.append(y)
#    all_features_test.append(test)
#    all_files.append(files)
#    all_nanbools.append(nanbool)
#np.savez('featuredata', train=all_features_train, y=all_y, test=all_features_test,
#         files=all_files, nanbools=all_nanbools)


# train
t = time.time()
np.random.seed(2016)
features_npz = np.load('featuredata.npz')
all_features_train, all_y, all_features_test, all_files, all_nanbools = features_npz['train'], features_npz['y'], features_npz['test'], features_npz['files'], features_npz['nanbools']
positives_SVC, positives_RF, positives_KNN = [], [], []
all_probs = {}
for train, y, test, files, nanbool in zip(all_features_train, all_y,
                                          all_features_test, all_files,
                                          all_nanbools):
#    # RF
#    Y_pred = train_RF(train, y, test)
#    print(str(int(sum(Y_pred))) + '/' + str(len(Y_pred)) + ' positives')
#    Y_sub = np.zeros(len(files),)
#    Y_sub[nanbool] = 0
#    Y_sub[~nanbool] = Y_pred
#    positives_RF += list(np.array(files)[Y_sub.astype(bool)])
#    print('RF train: ' + str(time.time() - t))

    # SVC
    Y_pred, Y_prob = train_SVC(train, y, test)
    print(str(int(sum(Y_pred))) + '/' + str(len(Y_pred)) + ' positives')
    Y_sub_pred = np.zeros(len(files),)
    Y_sub_pred[~nanbool] = Y_pred
    positives_SVC += list(np.array(files)[Y_sub_pred.astype(bool)])
    print('SVC train: ' + str(time.time() - t))

    Y_sub_prob = np.zeros(len(files),)
    Y_sub_prob[~nanbool] = Y_prob[:,1]
    prob_dict = dict(zip(files,Y_sub_prob))
    all_probs.update(prob_dict) 
#    try:
#        Y_sub = np.hstack((Y_sub,Y_sub_prob))
#    except:
#        Y_sub = Y_sub_prob
#    # KNN
#    Y_pred = train_KNN(train, y, test)
#    print(str(int(sum(Y_pred))) + '/' + str(len(Y_pred)) + ' positives')
#    Y_sub = np.zeros(len(files),)
#    Y_sub[nanbool] = 0
#    Y_sub[~nanbool] = Y_pred
#    positives_KNN += list(np.array(files)[Y_sub.astype(bool)])
#    print('KNN train: ' + str(time.time() - t))
    
sub = pd.read_csv('sample_submission.csv')
#sub['Class'] = [1 if x in positives_SVC else 0 for x in sub['File']]
sub['Class'] = [all_probs[x] for x in sub['File']]
#sub['Class'] = np.where(sub['File'] in positives, 1, 0)
#sub['Class'].apply(lambda x: 1 if sub['File'] in positives else 0)
#sub = pd.DataFrame({
#        "File": files,
#        "Class": Y_sub.astype(int)
#    })
#sub = sub[['File','Class', 'Prob']]
sub = sub[['File', 'Class']]
sub.to_csv('eeg.csv', index=False)


#for file in train_files:
##    print(file)
#    path = train_path + file
#    eeg = ieegMatToArray(path)
#    features = get_features(eeg)
#    ind = train_files.index(file)
#    if ind % 100 == 0:
#        print(round(ind/len_train*100))
#    feature_matrix[ind,:] = features
#    y[ind] = int(file[-5])
#nanbool = np.isnan(feature_matrix).any(axis=1)
#X_train = feature_matrix[~nanbool,:]
#y = y[~nanbool]
#print('Train: ' + str(sum(nanbool)) + ' NaN-rows removed')



#for file in test_files:
#    path = test_path + file
#    eeg = ieegMatToArray(path)
#    features = get_features(eeg)
#    ind = test_files.index(file)
#    if ind % 100 == 0:
#        print(round(ind/len_test*100))
#    feature_matrix_test[ind,:] = features
#nanbool = np.isnan(feature_matrix_test).any(axis=1)
#X_test = feature_matrix_test[~nanbool,:]
#print('Test: ' + str(sum(nanbool)) + ' NaN-rows removed')

#print(str(time.time() - t))

#svc = SVC()
#svc.fit(X_train, y)
#print(svc.score(X_train, y))
#Y_pred = svc.predict(X_test)



#interictal_testpath = traindata_path + "/1_101_0.mat"
#preictal_testpath = traindata_path + "/1_101_1.mat"
#ieegSingleMetaData(interictal_testpath)     
#
#x_ii=ieegMatToPandasDF(interictal_testpath)
#x_pi=ieegMatToPandasDF(preictal_testpath)
#matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)
#n=16
#for x in [x_ii,x_pi]:
#    for i in range(n):
#    #     print i
#        plt.subplot(n, 1, i + 1)
#        plt.plot(x[i +1])



#freq = np.fft.fft2(x)
#freq = np.abs(freq)
#print (freq)

#
#x_std = x.std(axis=1)
## print(x_std.shape, x_std.ndim)
#x_split = np.array(np.split(x_std, 100))
## print(x_split.shape)
#x_mean = np.mean(x_split, axis=0)
## print(x_mean.shape)
#plt.figure()
#plt.subplot(3, 1, 1)
#plt.plot(x)
#plt.subplot(3, 1, 2)
#plt.plot(x_std)
#plt.subplot(3, 1, 3)
#plt.plot(x_mean)



# NB. sequence (time position within the 1-hour window) not read atm
# SVC -> probability=True
# for matfile, extract and save features -> close
