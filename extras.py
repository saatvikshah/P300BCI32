import json
import numpy as np
import matplotlib.pyplot as plt

class Params:
    pass


def load_params():
    json_data = open("settings.json")
    data = json.load(json_data)
    params = Params()
    params.data_dir = data["data_dir"]
    params.fs = data["fs"]
    params.num_chars_train = data["num_chars_train"]
    params.num_chars_test = data["num_chars_test"]
    params.num_seq = data["num_seq"]
    return params




def visualize_avg(X,y,channel = 0):
    num_epochs,num_samples,num_channels = X.shape
    Xp300 = []
    Xnonp300 = []
    for i in xrange(num_epochs):
        if y[i] == 1:
            Xp300.append(X[i,:,channel])
        else:
            Xnonp300.append(X[i,:,channel])
    Xp300avg = np.sum(np.vstack(Xp300),axis = 0)
    Xnonp300avg = np.sum(np.vstack(Xnonp300),axis = 0)
    plt.plot(Xp300avg,"g",label="p300")
    plt.plot(Xnonp300avg,"r",label = "non p300")
    plt.legend()
    plt.show()



def visualize_tformed(X,y):
    Xp300 = []
    Xnonp300 = []
    for i in xrange(len(y)):
        if y[i] == 1:
            Xp300.append(X[i,:])
        else:
            Xnonp300.append(X[i,:])
    Xp300avg = np.mean(np.vstack(Xp300),axis=0)
    Xnonp300avg = np.mean(np.vstack(Xnonp300),axis=0)
    plt.plot(Xp300avg,"--go",label="p300")
    plt.plot(Xnonp300avg,"--ro",label = "non p300")
    plt.legend()
    plt.show()
