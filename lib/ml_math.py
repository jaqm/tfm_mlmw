#!/usr/bin/env python3

from math import log10

# TFIDF Criteria zone


def tfidf_function(tf, idf):
    tfidf = tf * idf
    return tfidf


def idf_function(num_samples, num_samples_with_this_word):
    '''IDF is the criteria of the Inverse Document Frequency.
    @params:
    num_samples: total amount of samples in the dataset.
    num_samples_with_this_word: amount of samples containing this word.
    '''

    idf = log10(num_samples/num_samples_with_this_word + 1)

    return idf


def tf_function(repetitions, words_in_the_document):
    '''TF is the amount of appearances of that word in a document
    (sample). Due to this, this value is different 
    for each word on each sample.
    @params:
    repetitions: amount of appearances of one word in this document.
    words_in_the_document: full amount fo words in the document.
    '''
    tf = repetitions / words_in_the_document
    return tf


def tfidf(
        word, repetitions, words_in_the_document,
        amount_of_samples, num_samples_with_this_word):

    tf = tf_function(repetitions,
                     words_in_the_document)
    idf = idf_function(amount_of_samples,
                       num_samples_with_this_word)
    tfidf = tfidf_function(tf, idf)
    return tfidf
