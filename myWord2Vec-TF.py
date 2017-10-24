"""
Using Tensorflow to implement my version of word2vec
"""

import sys
import collections

cmdLineArgs = sys.argv[1:]
dataFile = cmdLineArgs[0]

"""
Text Data obtained from http://mattmahoney.net/dc/text8.zip
"""
with open(dataFile) as dataf:
    lines = dataf.readlines()
    for line in lines:
        allWords = [word for word in line.split(" ")]
dataf.close()

print("Word Count in datafile: {}".format(len(allWords)))

"""Restricting Vocab to 100001 words including \"unknown\" word """
maxVocab = 100001

"""Building a Frequency Dictionary"""
freqDikt = [('UNK',-1)]
freqDikt.extend(collections.Counter(allWords).most_common(maxVocab -1))

"""Building a word index dictionary for maxVocab words"""
dikt = Dict()
for word, _ in freqDikt:
    dikt[word] = len(dikt)

transformedData = list()
for word in allWords:
    if word in dikt:
        transformedData.append(dikt[word])
    else:
        transformedData.append(dikt['UNK'])



