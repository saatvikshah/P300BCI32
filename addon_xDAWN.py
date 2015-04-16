__author__ = 'anirudha'
from scipy.signal import decimate
import scipy
import numpy as np



class BaseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

class xDWAN_filtering(BaseTransformer):



    def teoplitz_matrix(self,data,label):
        """
        label corresponding to the flashing(1 for target)
        run this code for all the data epochs
        """
        D1=np.array
        D=np.array
        #X=np.array
        for i in range(label.shape[0]):
            if label[i] == 1:
                D1 = np.diag(np.ones(data.shape[1]))
            else:
                D1 = np.zeros((data.shape[1], data.shape[1]))

            if i==0:
                X = np.array(data[i,:,:])
                D = D1
            else:
                X = np.vstack((X, data[i,:,:]))
                D = np.vstack((D, D1))


        print "Size of teoplitz matrix..."+str(D.shape)
        return X,D


    def generate_filter(self, X ,D):

        """
        Compute QR factorisation
        """
        # QR decompositions of X and D
        print X.shape
        Qx, Rx = np.linalg.qr(X)
            # QR decompositions of D
        Qd, Rd =  np.linalg.qr(D)

        """
        Compute SVD Qd.T Qx

        """
        print "Computing SVD..."
        Phi, Lambda, Psi = np.linalg.svd(np.dot(Qd.T, Qx),full_matrices=True)
        Psi = Psi.T
        print "SVD computed..."+str(Phi.shape)
        SNR = np.zeros(X.shape[1])
        #construct spatial filters
        for i in range(Psi.shape[1]):
            # Construct spatial filter with index i as Rx^-1*Psi_i
            ui = np.dot(np.linalg.inv(Rx), Psi[:,i])            #eq 12
            wi = np.dot(Rx.T, Psi[:,i])                         #eq 16
            if i < Phi.shape[1]:
                ai = np.dot(np.dot(np.linalg.inv(Rd), Phi[:,i]),Lambda[i])   #eq 15
            if i == 0:
                filters = np.atleast_2d(ui).T
                wi = np.atleast_2d(wi)
                ai = np.atleast_2d(ai)
            else:
                filters = np.hstack((filters,np.atleast_2d(ui).T))
                wi = np.vstack((wi, np.atleast_2d(wi)))
                if i < Phi.shape[1]:
                    ai = np.vstack((ai, np.atleast_2d(ai)))
            a = np.dot(D, ai.T)
            b = np.dot(X, ui)

            #SNR[i] = np.dot(a.T, a)/np.dot(b.T, b)
        print "Filters generated"
        return filters