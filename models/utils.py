#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def get_bleu_stats(ref, hyps):
    scores = []
    for hyp in hyps:
        try:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3,1./3]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5*tf.reduce_sum(tf.log(2*np.pi) + logvar + tf.div(tf.pow((x-mu), 2), tf.exp(logvar)), reduction_indices=1)


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z


def get_bow(embedding, avg=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    embedding_size = embedding.get_shape()[2].value
    if avg:
        return tf.reduce_mean(embedding, reduction_indices=[1]), embedding_size
    else:
        return tf.reduce_sum(embedding, reduction_indices=[1]), embedding_size


def get_rnn_encode(embedding, cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
        _, encoded_input = tf.nn.dynamic_rnn(cell, embedding, sequence_length=length_mask, dtype=tf.float32)
        return encoded_input, cell.state_size


def get_bi_rnn_encode(embedding, f_cell, b_cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
        _, encoded_input = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, embedding, sequence_length=length_mask, dtype=tf.float32)
        encoded_input = tf.concat(encoded_input, 1)
        return encoded_input, f_cell.state_size+b_cell.state_size
