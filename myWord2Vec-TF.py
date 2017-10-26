"""
Using Tensorflow to implement my version of word2vec
"""

import sys
import collections
import numpy as np
import random

cmdLineArgs = sys.argv[1:]
dataFile = cmdLineArgs[0]

"""
Text Data obtained from http://mattmahoney.net/dc/text8.zip
"""
with open(dataFile) as dataf:
    lines = dataf.readlines()
    for line in lines:
        allWords = [word for word in line.strip().split(" ")]
dataf.close()

print("Word Count in datafile: {}".format(len(allWords)))

"""Restricting Vocab to 100001 words including \"unknown\" word """
maxVocab = 100001

"""Building a Frequency Dictionary"""
freqDikt = [('UNK',-1)]
freqDikt.extend(collections.Counter(allWords).most_common(maxVocab -1))

"""Building a word index dictionary for maxVocab words"""
dikt = {}
for word, _ in freqDikt:
    dikt[word] = len(dikt)

transformedData = list()
for word in allWords:
    if word in dikt:
        transformedData.append(dikt[word])
    else:
        transformedData.append(dikt['UNK'])

print("Transformed Text: \"{}\" ==> {}".format(allWords[:5],transformedData[:5]))

"""
batch_size: Samples size before updating weights.
samples_per_window: Number of training samples to be generated for a given window.
skip_window: Number of neighbours to the left/right of the target. 
"""

data_index = 0

def get_next_batch(batch_size,samples_per_window,skip_window):

    global data_index

    batch = np.ndarray(shape=(batch_size))
    label = np.ndarray(shape=(batch_size,1))
    window_size = 2*skip_window + 1

    window = collections.deque(maxlen=window_size)

    if (data_index + window_size) > len(transformedData):
        data_index = 0

    window.extend(transformedData[data_index:window_size])
    data_index += window_size

    for i in range(batch_size/samples_per_window):

        potential_context_words = [s for s in range(window_size) if s != skip_window]
        words_to_use = random.sample(potential_context_words,samples_per_window)

        for j,context_word in enumerate(words_to_use):

            batch[i*samples_per_window + j] = window[skip_window]
            label[i*samples_per_window+j,0] = window[context_word]

        if data_index == len(transformedData):
            data_index = window_size
            window = transformedData[:window_size]
        else:
            window.append(transformedData[data_index])
            data_index +=1

    data_index = (data_index + len(transformedData) - window_size) % len(transformedData)

    return batch, label

"""
Test run for the get_next_batch
"""
batch,labels = get_next_batch(8,2,1)

for b,l in zip(batch,labels):
    print("Target Word: {} ".format(b)),
    print("Context Word: {}".format(l))











