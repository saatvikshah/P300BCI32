__author__ = 'tangy'
from batchio import *
from transforms import *
from filters import *
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.lda import LDA
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.metrics import confusion_matrix,accuracy_score
from classifier import *


def generate_model_params(params):
    ################    Define Parameters Here  ###############
    preprocessors = [
        [ButterworthFilter(sampling_rate=params.fs,fc1=0.1,fc2=15,input_type="epoched",order=4)],
    ]
    postprocessors = [
    ########### toppers
        [FeatureConcatenate(transform_list=[[ActualWaveletTransform(),EEGConcatExtracter(),Downsampler(rate=12,method="dec")],[WaveletTransform(),EEGConcatExtracter(),Downsampler(rate=12,method="dec")]])],
        [WaveletTransform(),EEGConcatExtracter(),Downsampler(rate=12,method="dec")],
        [FeatureConcatenate(transform_list=[[ActualWaveletTransform(),EEGConcatExtracter(),Downsampler(rate=24,method="dec")],[EEGConcatExtracter(),Downsampler(rate=24,method="dec")],[WaveletTransform(),EEGConcatExtracter(),Downsampler(rate=24,method="dec")]])],
        [FeatureConcatenate(transform_list=[[HjorthParams()],[EEGConcatExtracter(),Downsampler(rate=10,method="dec")]])],#-working
        [FeatureConcatenate(transform_list=[[EEGConcatExtracter(),Downsampler(rate=24)],[WaveletTransform(),EEGConcatExtracter(),Downsampler(rate=24)],[StatisticalFeatures()]])],
        [ActualWaveletTransform(),EEGConcatExtracter(),Downsampler(rate=12,method="dec")],
        [FeatureConcatenate(transform_list=[[EEGConcatExtracter(),Downsampler(rate=24,method="dec")],[WaveletTransform(),EEGConcatExtracter(),Downsampler(rate=24,method="dec")]])],
        [FeatureConcatenate(transform_list=[[FFTFeatures(),Downsampler(rate=12)],[EEGConcatExtracter(),Downsampler(rate=10,method="dec")]])],
        [FeatureConcatenate(transform_list=[[EEGConcatExtracter(),Downsampler(rate=24,method='dec')],[ActualWaveletTransform(),EEGConcatExtracter(),Downsampler(rate=24,method="dec")]])],
        [FeatureConcatenate(transform_list=[[EEGConcatExtracter(),Downsampler(rate=12)],[ActualWaveletTransform(),EEGConcatExtracter(),Downsampler(rate=12)],[HjorthParams()]])],
        [FeatureConcatenate(transform_list=[[EEGConcatExtracter(),Downsampler(rate=12)],[HjorthParams()],[SpectralEntropy(params)]])],
        [EEGConcatExtracter(),Downsampler(rate=12,method="dec")], #Benchmark - BCI32 winner -ok
        [FeatureConcatenate(transform_list=[[EEGConcatExtracter(),Downsampler(rate=12)],[ActualWaveletTransform(),EEGConcatExtracter(),Downsampler(rate=12)],[SpectralEntropy(params)],[HjorthParams()]])],
    ########## testing pending
    ]
    classifiers = [
        LinearSVC(C=0.001,loss='hinge',max_iter=3000,random_state=10),
        # LDA(),
        # BaggingClassifier(base_estimator=LDA(),n_estimators=500,n_jobs=-1,verbose=0),
        # BaggingClassifier(base_estimator=LinearSVC(C=0.001,loss='hinge',max_iter=3000),n_estimators=500,n_jobs=-1,verbose=0),
        # MODDEDLDA(),
        # RandomForestClassifier(n_estimators=1000),
    ]
    ##########################################################
    for preprocessor in preprocessors:
        for postprocessor in postprocessors:
            for classifier in classifiers:
                model_params = Params()
                model_params.preprocessor = TransformPipeline(preprocessor,params)
                model_params.postprocessor = TransformPipeline(postprocessor,params)
                model_params.clf = classifier
                yield model_params

def apply_model(Flashing,Signal,StimulusArr,params,model_params,type):
    preprocessor_pipe = model_params.preprocessor
    postprocessor_pipe = model_params.postprocessor
    clf = model_params.clf
    signal_preproc = preprocessor_pipe.transform(Signal,type)
    if type == "train":
        print "Training"
        Xtrain,ytrain = BCI32TrainEpochExtracter(params,offset_start=0,offset_end=0).transform(flashing_arr=Flashing,stimulus_arr=StimulusArr,signal_arr=signal_preproc)
        # visualize_avg(Xtrain,ytrain)
        # exit()
        del Flashing,Signal,StimulusArr
        Xtrain = postprocessor_pipe.transform(Xtrain,type,ytrain)
        # visualize_tformed(Xtrain,ytrain)
        clf.fit(Xtrain,ytrain)
        return clf
    else:
        print "Testing"
        X = BCI32TestEpochExtracter(params,offset_start=0,offset_end=0).transform(Flashing,signal_preproc)
        X = postprocessor_pipe.transform(X,type)
        try:
            ypred = clf.predict_proba(X)
        except Exception as e:
            ypred = clf.decision_function(X)
        if len(ypred.shape) > 1:
            #Probability of 1
            ypred = ypred[:,1]
        return ypred

def main():
    #sys.stdout = open("logs.txt","a")
    params = load_params()
    print "Subject A"
    print "Num of Seq : %s" % params.num_seq
    Flashing_TR,Signal_TR,StimulusType_TR,StimulusCode_TR = load_raw(subject="A",type="train")
    Flashing_TE,Signal_TE,StimulusCode_TE = load_raw(subject="A",type="test")
    for model_param in generate_model_params(params):
        print "Preprocessor %s" % str(model_param.preprocessor)
        print "Postprocessor %s" % str(model_param.postprocessor)
        print "Classifier %s" % str(model_param.clf)
        model_param.clf = apply_model(Flashing_TR,Signal_TR,StimulusType_TR,params,model_param,"train")
        ypred = apply_model(Flashing_TE,Signal_TE,None,params,model_param,"test")
        letters = PredictionToWordProba(params).transform(ypred,StimulusCode_TE)
        print "True Labels : " + str(get_true_labels(subject="A")[:params.num_chars_test])
        print "Predicted Labels : " + str(letters)
        print "Accuracy %f" % accuracy_score(get_true_labels(subject="A")[:params.num_chars_test],letters)




if __name__ == "__main__":
    main()

