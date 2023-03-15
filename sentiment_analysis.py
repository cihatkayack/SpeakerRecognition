import nltk
from os import getcwd
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sentiment_utils import process_tweet, count_tweets



def train_naive_bayes(freqs, train_x, train_y):

    loglikelihood = {}
    logprior = 0 
    
    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    
    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos=N_neg=V_pos=V_neg=0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            V_pos += 1
            N_pos += freqs[pair]
        # else, the label is negative
        else: 
            V_neg += 1
            N_neg += freqs[pair]

    # Calculate D, the number of documents
    D = train_y.shape[0]
    D_pos = train_y[train_y == 1].shape[0]
    D_neg = train_y[train_y == 0].shape[0]
    
    # Calculate logprior
    logprior  = np.log(D_pos / D) - np.log(D_neg / D)
    
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)
        
        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
     
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)


    return logprior, loglikelihood



def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
     p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)
    '''
    word_l = process_tweet(tweet)
    
    #initialize probability to zero
    p = 0
    p += logprior
    
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            p += loglikelihood[word]

    return p

def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):

    accuracy = 0

    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0

        y_hats.append(y_hat_i)

    
    accuracy = (np.sum(np.array(y_hats) == np.array(test_y)))/ len(y_hats)
    error = 1-accuracy
    # Accuracy is 1 minus the error


    return accuracy



nltk.download('twitter_samples')
nltk.download('stopwords')
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)
# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

freqs = count_tweets({}, train_x, train_y)



logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)
print(len(loglikelihood))

print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))


# for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
#     # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
#     p = naive_bayes_predict(tweet, logprior, loglikelihood)
# #     print(f'{tweet} -> {p:.2f} ({p_category})')
#     print(f'{tweet} -> {p:.2f}')
    



def predict_tweet_recognition(tweet,logprior=logprior,loglikelihood = loglikelihood, naive_bayes_predict=naive_bayes_predict):

    if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
        y_hat = 1
    else:
        y_hat = 0
    print(process_tweet(tweet))
    print(f"Result: {y_hat}")







