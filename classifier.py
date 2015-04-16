# __author__ = 'tangy'
#
# import numpy as np
# from pybrain.datasets import ClassificationDataSet
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.structure.modules import SoftmaxLayer
#
#
# class BackpropNN:
#
#     def __init__(self,num_epochs,hidden_layer_neurons,verbose = 1):
#         self.num_epochs = num_epochs
#         self.hidden_layer_neurons = hidden_layer_neurons
#         self.verbose = verbose
#
#     def fit(self,X,y):
#         num_samples,num_features = X.shape
#         self.num_labels = len(np.unique(y))
#         train_ds = ClassificationDataSet(num_features,nb_classes=self.num_labels)
#         for i in xrange(num_samples):
#             train_ds.addSample(X[i],y[i])
#         assert(len(y) == len(train_ds)),"Error Dataset lengths not matching"
#         train_ds._convertToOneOfMany()
#         self.clf = buildNetwork(train_ds.indim,self.hidden_layer_neurons,
#                                 train_ds.outdim,bias=True,
#                                 outclass=SoftmaxLayer)
#         self.trainer = BackpropTrainer(self.clf,train_ds)
#         for i in xrange(self.num_epochs):
#             err = self.trainer.train()
#             if self.verbose == 1: print "Epoch:%d Error:%f" % (i+1,err)
#
#
#     def predict(self,X):
#         num_samples,num_features = X.shape
#         y = []
#         for i in xrange(num_samples):
#             y.append(np.argmax(self.clf.activate(X[i])))
#         return np.array(y)
#
#     def predict_proba(self,X):
#         num_samples,num_features = X.shape
#         y = []
#         for i in xrange(num_samples):
#             y.append(self.clf.activate(X[i])[1])
#         return np.array(y)
#
#
#     def __repr__(self):
#         return ("BackpropNN__%d__%d" % (self.hidden_layer_neurons,self.num_epochs))
#
# class MODDEDLDA:
#
#     def fit(self,X,y):
#         """Train the LDA classifier.
#
#         Parameters
#         ----------
#         fv : ``Data`` object
#             the feature vector must have 2 dimensional data, the first
#             dimension being the class axis. The unique class labels must be
#             0 and 1 otherwise a ``ValueError`` will be raised.
#         shrink : Boolean, optional
#             use shrinkage
#
#         Returns
#         -------
#         w : 1d array
#         b : float
#
#         Raises
#         ------
#         ValueError : if the class labels are not exactly 0s and 1s
#
#
#         """
#         # TODO: this code should be trivially fixed to allow for two unique
#         # values instead of 0, 1 hardcoded
#         if not np.all(np.unique(y) == [0, 1]):
#             raise ValueError('Unique class labels are {labels}, should be [0, 1]'.format(labels=np.unique(y)))
#         mu1 = np.mean(X[y == 0], axis=0)
#         mu2 = np.mean(X[y == 1], axis=0)
#         # x' = x - m
#         m = np.empty(X.shape)
#         m[y == 0] = mu1
#         m[y == 1] = mu2
#         x2 = X - m
#         # w = cov(x)^-1(mu2 - mu1)
#         covm = np.cov(x2.T)
#         w = np.dot(np.linalg.pinv(covm), (mu2 - mu1))
#         # b = 1/2 x'(mu1 + mu2)
#         b = -0.5 * np.dot(w.T, (mu1 + mu2))
#         self.w = w
#         self.b = b
#
#     def predict_proba(self,X):
#         """Apply feature vector to LDA classifier.
#         """
#         return np.dot(X, self.w) + self.b
#
#     def __repr__(self):
#         return "ModdedLDA"
#
# class NLevelClassifier:
#
#     def __init__(self,classifier_pipe):
#         self.clf_pipe = classifier_pipe
#
#     def fit(self,X,y):
#        for clf_ind in xrange(len(self.clf_pipe) - 1):
#            self.clf_pipe[clf_ind].fit(X,y)
#            X = self.clf_pipe[clf_ind].decision_function(X)
#        if len(X.shape) < 2:
#             X = X.reshape(X.shape[0],1)
#        self.clf_pipe[-1].fit(X,y)
#        return self.clf_pipe
#
#     def predict(self,X):
#        for clf_ind in xrange(len(self.clf_pipe) - 1):
#            X = self.clf_pipe[clf_ind].decision_function(X)
#        if len(X.shape) < 2:
#             X = X.reshape(X.shape[0],1)
#        if "predict_proba" in dir(self.clf_pipe[-1]):
#            return self.clf_pipe[-1].predict_proba(X)[:,1]
#        else:
#            return self.clf_pipe[-1].predict(X)
#
#
