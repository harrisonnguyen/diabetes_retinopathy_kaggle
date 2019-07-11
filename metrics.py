import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(preds,actuals,N,eps=1e-10):

    """ Returns a value between 1 and -1. With 1 being optimal"""
    #return cohen_kappa_score(preds, actuals, weights='quadratic')
    repeat_op = tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N])
    repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
    weights = tf.cast(repeat_op_sq / (N - 1),tf.dtypes.float32)
    O = tf.math.confusion_matrix(actuals,preds,N,dtype=tf.dtypes.float32)
    act_hist = tf.math.bincount(actuals,minlength=N)
    pred_hist = tf.math.bincount(preds,minlength=N)
    E = tf.cast(tf.einsum('i,j->ij',act_hist,pred_hist),dtype=tf.dtypes.float32)
    O = O/tf.reduce_sum(O)
    E = E/tf.reduce_sum(E)
    num = tf.reduce_sum(tf.math.multiply(weights,O))
    den = tf.reduce_sum(tf.math.multiply(weights,E))

    return (1-(num/(den+eps)))

def sklearn_quadtratic_weighted_kappa(preds,actuals):
    return cohen_kappa_score(preds, actuals, weights='quadratic')
