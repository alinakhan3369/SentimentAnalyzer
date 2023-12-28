# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:12:01 2023

@author: alina
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# download the Brown Corpus and the VADER lexicon if you haven't already
nltk.download('brown')
nltk.download('vader_lexicon')

# import the Brown Corpus and select the "news" category
from nltk.corpus import brown

# create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# define a list of keywords for the issues or events you want to analyze
keywords = ['war', 'peace', 'economy', 'jobs', 'immigration']

# Get the speeches by Kennedy and Eisenhower
kennedy_speeches = ' '.join([w.lower() for w in brown.words(categories='news') if 'Kennedy' in w])
eisenhower_speeches = ' '.join([w.lower() for w in brown.words(categories='news') if 'Eisenhower' in w])

# analyze the sentiment of each speech and count the number of occurrences of each keyword
kennedy_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
eisenhower_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
for keyword in keywords:
    for sentence in kennedy_speeches.split('.'):
        if keyword in sentence:
            sentiment = sia.polarity_scores(sentence)['compound']
            if sentiment > 0:
                kennedy_sentiments['positive'] += 1
            elif sentiment < 0:
                kennedy_sentiments['negative'] += 1
            else:
                kennedy_sentiments['neutral'] += 1
    for sentence in eisenhower_speeches.split('.'):
        if keyword in sentence:
            sentiment = sia.polarity_scores(sentence)['compound']
            if sentiment > 0:
                eisenhower_sentiments['positive'] += 1
            elif sentiment < 0:
                eisenhower_sentiments['negative'] += 1
            else:
                eisenhower_sentiments['neutral'] += 1

# print the results
print('Sentiment analysis for Kennedy speeches:')
for k, v in kennedy_sentiments.items():
    print(k, ':', v)
print()
print('Sentiment analysis for Eisenhower speeches:')
for k, v in eisenhower_sentiments.items():
    print(k, ':', v)