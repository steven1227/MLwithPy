import pickle
import re
import nltk

__author__ = 'steven'
import os
import numpy as np
import scipy
from sklearn import svm, grid_search

with open(os.getcwd() + "/emailSample1.txt", 'r') as infile:
    sample = infile.read()

stemmer = nltk.stem.porter.PorterStemmer()

def processemail(email_contents,vocallist):
    word_indices = []
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']', email_contents)
    for token in tokens:
        token = re.sub( '[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token.strip())
        if len(token) == 0:
            continue
        if token in vocalist:
            word_indices.append(vocalist[token])
    return word_indices

def getemail_feature(indice):
    features = np.zeros((data.size,1))
    for i in indice:
        features[i, 0] = 1
    return features


data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + '/vocab.txt', delimiter='\n', dtype="string")
data = data.reshape((data.size, 1))

vocalist = {}
with open(os.getcwd() + "/vocab.txt", 'r') as infile:
    for temp in infile:
        temp2 = temp.rstrip('\n').split('\t')
        vocalist[temp2[1]] = temp2[0]

word_indices = processemail( sample, vocalist )
email_feature = getemail_feature(word_indices)

mat = scipy.io.loadmat( os.path.dirname(os.path.realpath(__file__))+"/spamTrain.mat" )
X, y = mat['X'], mat['y']
linear_svm = pickle.load( open("linear_svm.svm", "rb") )
