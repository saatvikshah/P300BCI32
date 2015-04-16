import numpy as np
import math
from sklearn.preprocessing import scale
from scipy.signal import decimate
import scipy
from joblib import Parallel,delayed
import pickle as pkl
import pywt
import pandas as pd
from pyeeg import hjorth,spectral_entropy,first_order_diff
import os


def store_pickle(data,path):
        f = open(path,"w")
        pkl.dump(data,f)
        f.close()

def read_pickle(path):
        f = open(path,"r")
        data = pkl.load(f)
        return data

####### Base Preprocessor

class BaseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

######  Extra Utilitites

class TransformPipeline:
    """
    Pure Transformer Pipeline
    """
    def __init__(self, transform_pipe, params):
        self.trans_pipe = transform_pipe
        self.params = params

    def transform(self, X, tag, y = None):
            path = os.path.join("./cache/",str(self.params.num_seq) + tag + str(self))
            if os.path.isfile(path):
                X = read_pickle(path)
                return X
            else:
                for tform in self.trans_pipe:
                    tform.fit(X,y)
                    X = tform.transform(X)
                store_pickle(X,path)
                return X

    def __repr__(self):
        name = str(self.params.num_chars_train)
        for tform in self.trans_pipe:
            name = '__'.join([name,str(tform)])
        return name



class FeatureConcatenate(BaseTransformer):

    """
        Feature Concatenate concatenates multiple transforms
        transform_list :
        a list for case of single transforms
        OR
        list of lists in case of concatenation of series of transforms
        ex. [FeatureConcatenate(transform_list=[[HjorthParams()],[EEGConcatExtracter(),Downsampler(rate=12,method="dec")]])],

    """

    def __init__(self,transform_list):
        self.transform_list = transform_list
        self.tl_type = any(isinstance(el, list) for el in self.transform_list)

    def transform(self, X):
        if self.tl_type is True:    #list of lists in case of concatenation of series of transforms
            Xtform = []
            for t_list in self.transform_list:
                X_t = X
                for t_func in t_list:
                    t_func.fit(X_t)
                    X_t = t_func.transform(X_t)
                Xtform.append(X_t)
            return np.hstack(Xtform)
        else:                       #a list for case of single transforms
            Xtform = []
            for tform in self.transform_list:
                tform.fit(X)
                Xtform.append(tform.transform(X))
            return np.hstack(Xtform)

    def __repr__(self):
        name = ""
        if self.tl_type is False:
            for tform in self.transform_list:
                name = '__'.join([name,str(tform)])
        else:
            for t_list in self.transform_list:
                for tform in t_list:
                    name = '__'.join([name,str(tform)])
        return name

######  Customized Train/Test Epoch Extracter

class BCI32TrainEpochExtracter:

    def __init__(self, params, offset_start=0,offset_end = 0):
        self.fs = params.fs
        self.num_chars = params.num_chars_train
        self.num_trials = params.num_seq
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.rc_count = 12

    def transform(self, flashing_arr, stimulus_arr, signal_arr):
        """
        :param flashing_arr: array containing information of of flashing
        :param stimulus_arr: array of desired flashes
        :param signal_arr: array of actual signal
        for extracting epochs and the y matrix from the signals
        """
        X = []
        y = []
        for i in range(self.num_chars):
            flash_indices = self.catch_flashes(flashing_arr[i, :])
            flash_indices = flash_indices[:self.num_trials*self.rc_count]
            p300_indices = self.catch_p300events(stimulus_arr[i, :])
            p300_indices = p300_indices[:self.num_trials*2]
            X.append(self.generate_x(flash_indices,signal_arr[i,:]))
            y.append(self.generate_y(flash_indices,p300_indices))
        Xall = np.vstack(X)
        yall = np.hstack(y)
        return Xall,yall

    def generate_y(self, flash_indices, p300_indices):
        """
        Generate the matrix y for comparison of machine learning
        :param flash_indices:
        :param p300_indices:
        :return:
        """
        y = np.zeros(len(flash_indices))
        count = 0
        for ind in range(len(y)):
            if count < len(p300_indices) and flash_indices[ind] == p300_indices[count]:
                count += 1
                y[ind] = 1
        return y

    def generate_x(self, flash_indices, signal_arr):
        """
        extract the epochs and append to 3 D array(flashes,signal,channels)

        """
        no_samples = math.ceil((700. / 1000 ) * self.fs)
        X = []
        for k in flash_indices:
            X.append(signal_arr[k + self.offset_start:k + no_samples - self.offset_end, :])
        return np.array(X)

    def catch_flashes(self, flashing):
        """
            Flashing is array of 1s and 0s where 1 corresponds to when there is a flash
            Here we catch the indices of this array wherever we get flashes to be starting
        """
        flash_indexes = []
        for i in xrange(flashing.shape[0]):
            if flashing[i] == 1 and flashing[i - 1] == 0:
                flash_indexes.append(i)
        return flash_indexes

    def catch_p300events(self, stimulus):
        """
            Flashing is array of 1s and 0s where 1 corresponds to when there is a flash
            Here we catch the indices of this array wherever we get flashes to be starting
            """
        p300_indexes = []
        for i in xrange(stimulus.shape[0]):
            if stimulus[i] == 1 and stimulus[i - 1] == 0:
                p300_indexes.append(i)
        return p300_indexes

    def __repr__(self):
        return "BCI32TrainEpochExtracter"

class BCI32TestEpochExtracter:

    def __init__(self, params, offset_start=0,offset_end = 0):
        self.fs = params.fs
        self.num_chars = params.num_chars_test
        self.num_trials = params.num_seq
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.rc_count = 12

    def transform(self, flashing_arr, signal_arr):
        """

        :param flashing_arr: array containing information of of flashing
        :param stimulus_arr: array of desired flashes
        :param signal_arr: array of actual signal
        for extracting epochs and the y matrix from the signals
        """
        X = []
        for i in range(self.num_chars):
            flash_indices = self.catch_flashes(flashing_arr[i, :])
            flash_indices = flash_indices[:self.num_trials*self.rc_count]
            X.append(self.generate_x(flash_indices, signal_arr[i, :]))
        return np.vstack(X)

    def generate_x(self, flash_indices, signal_arr):
        """
        extract the epochs and append to 3 D array(flashes,signal,channels)

        """
        no_samples = math.ceil((700. / 1000 ) * self.fs)
        X = []
        for k in flash_indices:
            X.append(signal_arr[k + self.offset_start:k + no_samples - self.offset_end, :])
        return np.array(X)

    def catch_flashes(self, flashing):
        """
            Flashing is array of 1s and 0s where 1 corresponds to when there is a flash
            Here we catch the indices of this array wherever we get flashes to be starting
        """
        flash_indexes = []
        for i in xrange(flashing.shape[0]):
            if flashing[i] == 1 and flashing[i - 1] == 0:
                flash_indexes.append(i)
        return flash_indexes

    def __repr__(self):
        return "BCI32TestEpochExtracter"

#####   Customized Predictors

class PredictionToWord:

    def __init__(self,params):
        self.num_letters = params.num_chars_test
        self.flashes_per_trial = 12
        self.num_trials_per_letter = params.num_seq

    def transform(self,ypred,rowcol_arr,flashing_arr):
        self.flashes_per_char = self.flashes_per_trial*self.num_trials_per_letter
        voter = [[0 for i in xrange(self.flashes_per_trial)] for i in xrange(self.num_letters)]
        pred_letter = []
        for letter_ind in xrange(self.num_letters):
            letter_flash_indices = self.catch_flashes(flashing_arr[letter_ind])
            assert len(letter_flash_indices) == (len(ypred)/self.num_letters) == \
                    self.flashes_per_trial*self.num_trials_per_letter,'Index Lengths not matching up'
            for trial in xrange(self.num_trials_per_letter):
                for rc_int_ind in xrange(self.flashes_per_trial):
                    flash_ind = trial*self.flashes_per_trial + rc_int_ind
                    if ypred[flash_ind] == 1:
                        rowcol_ind = int(rowcol_arr[letter_ind,letter_flash_indices[flash_ind]]) - 1
                        voter[letter_ind][rowcol_ind] += 1
            col_ind = np.argmax(voter[letter_ind][:self.flashes_per_trial/2])
            row_ind = np.argmax(voter[letter_ind][self.flashes_per_trial/2:self.flashes_per_trial])
            pred_letter.append(self.get_word(row_ind,col_ind))
        return pred_letter


    def get_word(self,row_ind,col_ind):
        word_matrix=['ABCDEF','GHIJKL','MNOPQR','STUVWX','YZ1234','56789_']
        return word_matrix[row_ind][col_ind]

    def catch_rowcolindices(self, rowcol_arr, flash_indices):
        rc_indices = []
        for char in xrange(self.num_letters):
            for ind in flash_indices:
                rc_indices.append(rowcol_arr[char,ind])
        return rc_indices

    def catch_flashes(self, flashing):
        """
            Flashing is array of 1s and 0s where 1 corresponds to when there is a flash
            Here we catch the indices of this array wherever we get flashes to be starting
        """
        flash_indexes = []
        for i in xrange(flashing.shape[0]):
            if flashing[i] == 1 and flashing[i - 1] == 0:
                flash_indexes.append(i)
        return flash_indexes

class PredictionToWordProba:

    def __init__(self,params):
        self.num_chars = params.num_chars_test
        self.num_seq = params.num_seq
        self.total_seq = 15
        self.num_rc = 12

    def transform(self,ypred,rowcol_arr):
        MATRIX = ['abcdef',
                  'ghijkl',
                  'mnopqr',
                  'stuvwx',
                  'yz1234',
                  '56789_']
        flattened_rc_arr = np.reshape(rowcol_arr[:self.num_chars],-1)
        rowcol_ind = self.duplicate_killer(flattened_rc_arr)
        rowcol_ind = self.extract_seq(rowcol_ind)
        #rowcol_ind = rowcol_ind.reshape(self.num_chars,self.num_seq,self.num_rc)
        unscramble_idx = rowcol_ind.argsort()
        static_idx = np.indices(unscramble_idx.shape)
        lda_out_prob = ypred.reshape(self.num_chars, self.num_seq, self.num_rc)
        lda_out_prob = lda_out_prob[static_idx[0], static_idx[1], unscramble_idx]
        lda_out_prob = lda_out_prob.sum(axis=1)
        lda_out_prob = lda_out_prob.argsort()
        cols = lda_out_prob[lda_out_prob <= 5].reshape(self.num_chars, -1)
        rows = lda_out_prob[lda_out_prob > 5].reshape(self.num_chars, -1)
        text = ''
        for i in range(self.num_chars):
            row = rows[i][-1]-6
            col = cols[i][-1]
            letter = MATRIX[row][col]
            text += letter.upper()
        return list(text)

    def extract_seq(self,rc_arr):
        rc_arr = rc_arr.reshape(self.num_chars,self.total_seq,self.num_rc)
        rc_arr = rc_arr[:self.num_chars,:self.num_seq,:].reshape(self.num_chars,self.num_seq,self.num_rc)
        return rc_arr

    def duplicate_killer(self,flat_nz_rc_arr):
        uniq_flat_rcarr = []
        for i in xrange(flat_nz_rc_arr.shape[0]):
            if flat_nz_rc_arr[i] != 0 and flat_nz_rc_arr[i - 1] == 0:
                uniq_flat_rcarr.append(flat_nz_rc_arr[i])
        return np.array(uniq_flat_rcarr,dtype=int)

######  Preprocessors

class RejectChannel(BaseTransformer):
    """
    Reject specific channels
    """

    def __init__(self, toreject, num_channels=64):
        all = range(num_channels)
        self.toreject = toreject
        self.tokeep = list(set(all) - set(toreject))


    def transform(self, X):
        return X[:, :, self.tokeep]

    def __repr__(self):
        return "RejectChannel_" + str(self.toreject)

class KeepChannel(BaseTransformer):
    """
    Reject specific channels
    """

    def __init__(self, tokeep, num_channels=64):
        self.tokeep = tokeep


    def transform(self, X):
        return X[:, :, self.tokeep]

    def __repr__(self):
        return "KeepChannel_" + str(self.tokeep)

######  Postprocessors

class EEGConcatExtracter(BaseTransformer):
    def __init__(self):
        pass

    def transform(self, X):
        n_epochs, n_samples, n_channels = X.shape
        X_tf = np.zeros((n_epochs, n_channels * n_samples))
        for epoch in xrange(n_epochs):
            for channel in xrange(n_channels):
                X_tf[epoch, channel * n_samples:(channel + 1) * n_samples] = X[epoch, :, channel]
        return X_tf

    def __repr__(self):
        return '_'.join(["EEGConcatExtracter"])

##  3d input -> 3d output/Requiring EEGConcatExtractor to create feature vector

def apply_wt(X,cutoff_freq):
    n_samples, n_channels = X.shape
    b,a = scipy.signal.cheby1(4,0.5,cutoff_freq,btype='low')
    XWT2D = []
    for j in xrange(n_channels):
        Xchsamples = X[:,j]
        tformed=[]
        Xfilt = scipy.signal.filtfilt(b,a,Xchsamples)
        n_samples=len(Xfilt)
        while(n_samples>21):
            Xds = scipy.signal.resample(Xfilt,math.floor(n_samples/2))
            n_samples=len(Xds)
            b,a = scipy.signal.cheby1(4,0.5,cutoff_freq/2,btype='low')
            Xfiltds = scipy.signal.filtfilt(b,a,Xds)
            tformed.append(Xfiltds)
        XWT2D.append(np.hstack(tformed))
    return np.vstack(XWT2D).T

class WaveletTransform(BaseTransformer):

    #added by Ani

    def __init__(self,fs=247.0,fc=47.0):
        self.fs=fs
        self.fc=fc


    def transform(self,X):
        n_epochs, n_samples, n_channels = X.shape
        cutoff_freq = 2*self.fc/self.fs
        # Parallel Test
        XWT2d = np.array(Parallel(n_jobs=-1,verbose=2)(delayed(apply_wt)(X[k],cutoff_freq)
                                    for k in xrange(n_epochs)))
        #print(np.mean(XWT2d,axis=0))
        #mean=np.mean(XWT2d,axis=0)
        #print mean.shape
        return XWT2d

    def __repr__(self):
        return "WaveletTransform"

class ActualWaveletTransform(BaseTransformer):
    """
    Apply Wavelet transform on Epoched EEG Data
    Get the multilevel approx. coefficients
    """
    def __init__(self):
        self.wvlt = pywt.Wavelet('db2')

    def transform(self, X):
        n_epochs,n_samples,n_channels = X.shape
        Xwvtf = []
        for epoch in xrange(n_epochs):
            Xepwvtf = []
            for chan in xrange(n_channels):
                Xepwvtf.append(pywt.dwt(X[epoch,:,chan],self.wvlt)[0])
            Xwvtf.append(Xepwvtf)
        Xwvtf = np.array(Xwvtf)
        n_epochs,n_channels,n_samples = Xwvtf.shape
        Xwvtf = Xwvtf.reshape(n_epochs,n_samples,n_channels)
        return Xwvtf

    def __repr__(self):
        return "ActualWaveletTransform"
##  3d input -> 2d output

#   pending
class HjorthParams(BaseTransformer):
    """ Hjorth parameters are statistical properties in Signals
        - activity,mobility,complexity
        activity : represents signal power = variance of signal with time
        mobility : mean freq or proportion of standard deviation of the power spectrum
        complexity : represents changes in frequency
        Used in analysis of EEG Signals for feature extraction

    """

    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        Xhjorth = []
        for epoch in xrange(num_epochs):
            xch = []
            for chan in xrange(num_channels):
                hjparams = hjorth(X[epoch,:,chan])
                xch.append(hjparams[0])
                xch.append(hjparams[1])
            Xhjorth.append(xch)
        Xtf = np.vstack(Xhjorth)
        return Xtf

    def __repr__(self):
        return "HjorthParams"

#   pending
class SpectralEntropy(BaseTransformer):

    def __init__(self,params):
        self.fs = params.fs

    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        Xhjorth = []
        for epoch in xrange(num_epochs):
            xch = []
            for chan in xrange(num_channels):
                xch.append(spectral_entropy(X[epoch,:,chan],[0.5,4,7,12,30],self.fs))
            Xhjorth.append(xch)
        Xtf = np.vstack(Xhjorth)
        print Xtf.shape
        return Xtf

    def __repr__(self):
        return "Spectral Entropy"

#   pending
class FFTFeatures(BaseTransformer):
    def __init__(self):
        pass

    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        Xfft = []
        for i in xrange(num_epochs):
            x = []
            for j in xrange(num_channels):
                x.append(np.log10(np.absolute(np.fft.rfft(X[i,:,j])))[2:30])
            Xfft.append(np.hstack(x))
        Xfft = np.vstack(Xfft)
        return Xfft

    def __repr__(self):
        return "FFTSlice"

#   pending
class TimeSeriesDifferential(BaseTransformer):
    """ First Order Differential of Time Series Data
    """

    def __repr__(self):
        return "TimeSeriesDataDifferentiated"

    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        Xtf = []
        for ep in xrange(num_epochs):
            x = []
            for ch in xrange(num_channels):
                x.append(first_order_diff(X[ep,:,ch]))
            Xtf.append(np.hstack(x))
        Xtf = np.vstack(Xtf)
        return Xtf

#   pending
class StatisticalFeatures(BaseTransformer):
    """ Computing simple statistical features from Time Series Data
    1. Mean
    2. Standard Deviation
    3. Skewness : Measurement of lack of symmetry
                (will be 0-unskewed,1-positively skewed,-1-negatively skewed)
    4. Kurtosis : Degree to which distribution is peaked(+ve)
    4. Max and Min
    """

    def __repr__(self):
        return "Simple_Statistical_Features"

    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        Xtf = []
        for ep in xrange(num_epochs):
            xfeats = []
            for chan in xrange(num_channels):
                xfeats.append(np.mean(X[ep,:,chan]))    #mean
                xfeats.append(np.std(X[ep,:,chan]))     #std
                xfeats.append(np.max(X[ep,:,chan]))     #max
                xfeats.append(np.min(X[ep,:,chan]))     #min
                xfeats.append(scipy.stats.skew(X[ep,:,chan]))   #skewness
                xfeats.append(scipy.stats.kurtosis(X[ep,:,chan],bias=False))   #kurtosis
            Xtf.append(np.hstack(xfeats))
        return np.vstack(Xtf)

#   pending
class FFTPeaks(BaseTransformer):
    """ Frequencies of k peaks in amplitude derived from FFT
    """

    def __init__(self,num_peaks):
        self.num_peaks = num_peaks

    def transform(self, X):
        num_epochs,num_samples,num_channels = X.shape
        Xtf = []
        for ep in xrange(num_epochs):
            xfeat = []
            for chan in xrange(num_channels):
                fftfeat = np.log10(np.absolute(np.fft.rfft(X[ep,:,chan])))
                xfeat.append(fftfeat[np.argsort(fftfeat)[-self.num_peaks:][::-1]])
            Xtf.append(np.hstack(xfeat))
        return np.vstack(Xtf)

    def __repr__(self):
        return "%s_%s" % ("FFTPeaks",self.num_peaks)

##  2d input -> 2d output

class Downsampler(BaseTransformer):
    """
    Downsample by
    -Averaging
    -Decimation
    """

    def __init__(self, method="dec", rate=10):
        """
        :param method:"avg" for averaging,"dec" for decimation
        :param rate: downsampling rate
        """
        self.method = method
        self.rate = rate

    def transform(self, X):
        if self.method == "avg":
            return self.averager(X)
        elif self.method == "dec":
            return self.decimater(X)

    def averager(self, X):
        n_epochs, n_features = X.shape
        Xavg = np.zeros((n_epochs, int(n_features / self.rate)))
        for ep in xrange(n_epochs):
            for feat in xrange(int(n_features / self.rate)):
                Xavg[ep, feat] = np.mean(X[ep, feat * self.rate:feat * self.rate + self.rate])
        return Xavg


    def decimater(self, X):
        n_epochs, n_features = X.shape
        X_ds = []
        for ep in xrange(n_epochs):
            X_ds.append(self.direct_downsampler(X[ep, :]))
        return np.array(X_ds)

    def direct_downsampler(self,Xep):
        num_features = Xep.shape[0]
        Xds = []
        for i in xrange(num_features):
            if i % self.rate == 0:
                Xds.append(Xep[i])
        return Xds

    def __repr__(self):
        return '_'.join(["Downsampler", str(self.method), str(self.rate)])

### data input independent -> 2d

class P300Trial_SequenceConcantenator(BaseTransformer):
    """
    for concatenating P300 trial and Sequence as a feature
    """
    def __init__(self,num_seq):
        self.num_seq = num_seq

    def transform(self, X):
        n_flashes_per_trial = 12
        n_trials = self.num_seq
        n_samples,n_epoch,n_channel = X.shape
        n_char = n_samples/(n_trials*n_flashes_per_trial)
        Xtrain = np.zeros((n_samples,3))  # 3 cols : char,trial
        for i in range(n_char):
            Xtrain[i*n_trials*n_flashes_per_trial:(i+1)*n_trials*n_flashes_per_trial,0] = i
            for j in range(n_trials):
                Xtrain[i*n_trials*n_flashes_per_trial + j*n_flashes_per_trial:i*n_trials*n_flashes_per_trial + (j+1)*n_flashes_per_trial,1] = j
        return Xtrain

    def __repr__(self):
        return "P300Trial_SequenceConcantnator"

### under construction

class Spearman_Correlation(BaseTransformer):
    """
    for removing highly correlated features
    """
    def __init__(self):
        pass

    def transform(self, X):
        """
        # Use Spearman correlation to remove highly correlated features

        # calculate the correlation matrix
        """
        n_epoch,n_samples,n_channel = X.shape
        X_modified=[]
        for i in range(n_epoch):
            flag = X[i,:,:]
            #flag = flag.T
            df=pd.DataFrame(flag)
            df_corr = df.corr(method='spearman')

            # create a mask to ignore self-
            mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
            df_corr = mask * df_corr

            drops = []
            # loop through each variable
            for col in df_corr.columns.values:
                # if we've already determined to drop the current variable, continue
                if np.in1d([col],drops):
                    continue

                # find all the variables that are highly correlated with the current variable
                # and add them to the drop list
                corr = df_corr[abs(df_corr[col]) > 0.98].index
                #print col, "highly correlated with:", corr
                drops = np.union1d(drops, corr)

            print "\nDropping", drops.shape[0], "highly correlated features...\n" #, drops
            df.drop(drops, axis=1, inplace=True)
            #flag = df.values
            if i==0:
                X_flag = df.values
                x,y= X_flag.shape
                X_modified = np.zeros((x,y))
            X_modified = np.dstack((X_modified,df.values))
        return X_modified[1:,:,:]



    #*********************************************************************************************************

    def __repr__(self):
        return "Spearman_Correlation"

### end

## fuck theese?
class FreqEigNCoeff(BaseTransformer):
    def __init__(self, slice_index):
        self.slice_index = slice_index

    def transform(self, X):
        n_epochs, n_samples, n_channels = X.shape
        Xfft = []
        for ep in xrange(n_epochs):
            xfft = []
            for chan in xrange(n_channels):
                xfft.append(np.log10(np.abs(np.fft.rfft(X[ep, :, chan], axis=0)[1:self.slice_index])))
            # Here samples per channel are features
            scaled = scale(np.vstack(xfft).T, axis=0)
            corr_matrix = np.corrcoef(scaled)
            eigenvalues = np.abs(np.linalg.eig(corr_matrix)[0])
            eigenvalues.sort()
            corr_coefficients = self.upper_right_triangle(corr_matrix)
            Xfft.append(np.concatenate((corr_coefficients, eigenvalues)))
        return np.vstack(Xfft)

    def upper_right_triangle(self, corr_matrix):
        num_d1, num_d2 = corr_matrix.shape
        coeff = []
        for i in xrange(num_d1):
            coeff.append(corr_matrix[i, i:])
        return np.hstack(coeff)

    def __repr__(self):
        return "FreqEigNCoeff_" + str(self.slice_index)

class TimeEigNCoeff(BaseTransformer):
    """
    Finds Time Eigenvalues and Flattened Correlation Matrix
    """

    def transform(self, X):
        n_epochs, n_samples, n_features = X.shape
        Xtime = []
        for ep in xrange(n_epochs):
            scaled = scale(np.vstack(X[ep].T), axis=0)
            corr_matrix = np.corrcoef(scaled)
            eigenvalues = np.abs(np.linalg.eig(corr_matrix)[0])
            eigenvalues.sort()
            corr_coefficients = self.upper_right_triangle(corr_matrix)
            Xtime.append(np.concatenate((corr_coefficients, eigenvalues)))
        return np.vstack(Xtime)

    def upper_right_triangle(self, corr_matrix):
        num_d1, num_d2 = corr_matrix.shape
        coeff = []
        for i in xrange(num_d1):
            coeff.append(corr_matrix[i, i:])
        return np.hstack(coeff)

    def __repr__(self):
        return "TimeEigNCoeff_"

class xDAWN_filtering(BaseTransformer):

    def fit(self, X, y):
        if y is not None:
            X,D = self.generate_teoplitz_matrix(X,y)
            self.generate_filter(X,D)
        else:
            pass

    def transform(self, X):
        n_epochs,n_samples,n_channels=X.shape
        Xflat = X.reshape((n_epochs*n_samples,n_channels))
        filters = self.filters_load()
        projected_data = np.dot(Xflat,filters)
        del Xflat,filters
        """
        Now make the data into epochs
        """
        xDAWN=np.zeros((n_samples,n_channels))
        for i in range(n_epochs):
            xDAWN= np.dstack((xDAWN,projected_data[i*n_samples:(i+1)*n_samples,:]))
        return xDAWN[:,:,1:].T


    def generate_teoplitz_matrix(self,X,y):
        """
        label corresponding to the flashing(1 for target)
        run this code for all the data epochs
        """
        n_epochs,n_samples,n_channels = X.shape
        Xflat = X.reshape((n_epochs*n_samples,n_channels))
        for i in xrange(n_epochs):
            if y[i] == 1:
                D1 = np.diag(np.ones(n_samples))
            else:
                D1 = np.zeros((n_samples, n_samples))

            if i==0:
                D = D1
            else:
                D = np.vstack((D, D1))
        return Xflat,D


    def generate_filter(self, X ,D):

        """
        Compute QR factorisation
        """
        # QR decompositions of X and D
        Qx, Rx = np.linalg.qr(X)
            # QR decompositions of D
        Qd, Rd =  np.linalg.qr(D)

        """
        Compute SVD Qd.T Qx

        """
        Phi, Lambda, Psi = np.linalg.svd(np.dot(Qd.T, Qx),full_matrices=True)
        Psi = Psi.T
        #construct spatial filters
        for i in range(Psi.shape[1]):
            # Construct spatial filter with index i as Rx^-1*Psi_i
            ui = np.dot(np.linalg.inv(Rx), Psi[:,i])            #eq 12
            if i < Phi.shape[1]:
                ai = np.dot(np.dot(np.linalg.inv(Rd), Phi[:,i]),Lambda[i])   #eq 15
            if i == 0:
                filters = np.atleast_2d(ui).T
                ai = np.atleast_2d(ai)
            else:
                filters = np.hstack((filters,np.atleast_2d(ui).T))
                if i < Phi.shape[1]:
                    ai = np.vstack((ai, np.atleast_2d(ai)))
        self.filters_dump(filters)
        return filters

    def filters_dump(self,obj):
        f = open("filters.temp","wb")
        pkl.dump(obj,f)
        f.close()

    def filters_load(self):
        f = open("filters.temp","r")
        obj = pkl.load(f)
        f.close()
        return obj

    def __repr__(self):
        return "xDAWN_filtering"

