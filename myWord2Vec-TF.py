"""
Using Tensorflow to implement my version of word2vec
"""

import sys
import collections
import numpy as np
import random
import tensorflow as tf
import math

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

    window.extend(transformedData[data_index:data_index+window_size])
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

"""
Reset the data index back to 0
"""
data_index = 0

"""
The TF Graph Setup 
"""
batch_size = 128
embed_dimension = 128
skip_window = 1
samples_per_window = 2
num_sampled = 64

graph = tf.Graph()

with graph.as_default():

    """Define Inputs & Outputs"""
    train_x = tf.placeholder(tf.int32,shape=[batch_size],name="inputs")
    train_y = tf.placeholder(tf.int32,shape=[batch_size,1],name="labels")

    embeddings = tf.Variable(tf.random_uniform([maxVocab,embed_dimension],minval=-1.0,maxval=1.0))
    embedding_lookup = tf.nn.embedding_lookup(embeddings,train_x)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),axis=1,keep_dims=True))
    embedding_norm = embeddings/norm

    weights = tf.Variable(tf.truncated_normal([maxVocab,embed_dimension],stddev=1/math.sqrt(maxVocab)))
    biases = tf.Variable(tf.zeros(maxVocab))

    """Noise Contrastive Training Objective"""

    nce_loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=weights,biases=biases,
                                   labels=train_y,inputs=embedding_lookup,
                                   num_sampled=num_sampled,num_classes=maxVocab))

    """SGD Optimizer"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(nce_loss)

    tf_init = tf.global_variables_initializer()

num_steps = 100000

print("Start Training")

with tf.Session(graph=graph) as session:

    tf_init.run()

    average_loss = 0

    for step in xrange(num_steps):
        batch, label = get_next_batch(batch_size,samples_per_window,skip_window)

        data_dikt = {train_x:batch, train_y:label}

        _,loss = session.run([optimizer,nce_loss],feed_dict=data_dikt)
        average_loss += loss

        if step % 1000 == 0:
            print("Average Loss after {} iteration: {}".format(step,average_loss/1000))
            average_loss = 0

    final_embeddings = embedding_norm.eval()








