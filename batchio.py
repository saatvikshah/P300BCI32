import os
import scipy.io as scio
from extras import *
import numpy as np
from sklearn.cross_validation import cross_val_score,KFold
from sklearn.metrics import confusion_matrix,accuracy_score

def load_raw(subject="A",type="train"):
    """
    load raw data according to parameters passed
    :param subject:"A"/"B"
    :param type:"train"/"test"
    :return:Flashing,Signal,StimulusType
    """
    params = load_params()
    for subdir,dirs,files in os.walk(params.data_dir):
        for file in files:
            definers = file.strip("mat").strip(".").split("_")
            if len(definers) == 3 and definers[1] == subject and definers[2].lower() == type:
                mat = scio.loadmat(os.path.join(params.data_dir,file))
    Flashing = mat["Flashing"]
    Signal = mat["Signal"]
    StimulusCode = mat["StimulusCode"]
    if type == "train":
        StimulusType = mat["StimulusType"]
        return Flashing,Signal,StimulusType,StimulusCode
    else:
        return Flashing,Signal,StimulusCode

def train_test_batches(X,y,test_ratio):
    full_size = len(y)
    test_size = int(full_size*test_ratio)
    midpt = full_size/2
    test_range = range(midpt - test_size/2,midpt + test_size/2)
    train_range = list(set(range(full_size)) - set(test_range))
    return X[train_range],X[test_range],y[train_range],y[test_range]

def cv_batches(Xp300,Xnonp300,yp300,ynonp300,test_ratio=0.3,smart_extract=False,shuffling=False):
    if smart_extract:
        Xp300train,Xp300test,yp300train,yp300test = \
            train_test_batches(Xp300,yp300,test_ratio)
        Xnop300train,Xnop300test,ynop300train,ynop300test = \
            train_test_batches(Xnonp300,ynonp300,test_ratio)
        Xtrain =  np.concatenate((Xp300train,Xnop300train))
        Xtest =  np.concatenate((Xp300test,Xnop300test))
        ytrain =  np.concatenate((yp300train,ynop300train))
        ytest =  np.concatenate((yp300test,ynop300test))
    else:
        X = np.concatenate((Xp300,Xnonp300))
        y = np.concatenate((yp300,ynonp300))
        xy_tuples = zip(X,y)
        del X,y
        if shuffling:
            np.random.seed(15)
            np.random.shuffle(xy_tuples)
        split_pt = int(len(xy_tuples) - len(xy_tuples)*test_ratio)
        train_indices = range(split_pt)
        test_indices = range(split_pt,len(xy_tuples))
        Xtrain,ytrain = (np.array(l) for l in zip(*[xy_tuples[i] for i in train_indices]))
        try:
            Xtest,ytest = (np.array(l) for l in zip(*[xy_tuples[i] for i in test_indices]))
        except:
            Xtest = []
            ytest = []
    return Xtrain,Xtest,ytrain,ytest

def cross_validate_cm(clf,X,y,cv=4):
    kf = KFold(n=len(y),n_folds=cv,shuffle=True,random_state=42)
    for train_indices,test_indices in kf:
        X_train,y_train = X[train_indices],y[train_indices]
        X_test,y_test = X[test_indices],y[test_indices]
        clf.fit(X_train,y_train)
        ypred = clf.predict(X_test)
        print confusion_matrix(y_test,ypred)
        print accuracy_score(y_test,ypred)

def get_true_labels(subject="A"):
    params = load_params()
    for subdir,dirs,files in os.walk(params.data_dir):
        for file in files:
            definers = file.strip(".txt").split("_")
            if len(definers) == 2 and definers[0] == subject:
                return list(open(os.path.join(params.data_dir,file),"r").readline())