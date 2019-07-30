import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout,Lambda
from keras import backend as K
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

from sklearn.model_selection import train_test_split
from metrics import quadratic_weighted_kappa,sklearn_quadratic_weighted_kappa
# create the base pre-trained model
import preprocess
import click
import os
import pandas as pd
import pathlib
import numpy as np
from glob import glob
import os
import math

def step_decay(epoch, lr):
    # initial_lrate = 1.0 # no longer needed
    drop = 0.9
    epochs_drop = 1.0
    lrate = lr * math.pow(drop,
    math.floor((1+epoch)/epochs_drop))
    return lrate

def argmax_layer(x):
    return tf.cast(tf.keras.backend.argmax(x,axis=-1),tf.float32)

def create_model():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2))(x) #1e-4, 1e-4
    x = Dropout(rate=0.5)(x)
    # and a logistic layer
    predictions = Dense(5, activation='softmax',name='softmax')(x)
    arg_predictions = Lambda(argmax_layer,name="argmax")(predictions)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=[arg_predictions,predictions])
    optimiser = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.1,amsgrad=False)
    #model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy',
    #            metrics=['accuracy'])
    loss_func ={
        'argmax':"mse",
    }
    metrics={
        'softmax':['sparse_categorical_crossentropy','accuracy'],
        'argmax': ['mse']
    }
    model.compile(optimizer=optimiser, loss=loss_func,
                metrics=metrics)
    return model,base_model
@click.command()
@click.option('--batch-size',
                default=8,
                type=click.INT,
                help="Batch size",
                show_default=True)
@click.option('--training-dir',
                default='dataset/output_combined2',
                type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
                help="dir for training data",
                show_default=True)
@click.option('--checkpoint-dir',
                default='/home/harrison/tensorflow_checkpoints/diabetes/',
                type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
                help="dir checkpoints",
                show_default=True)
@click.option('--epochs',
                default=2,
                type=click.INT,
                help="Number of epochs to train",
                show_default=True)
@click.option('--n-fixed-layers',
                default=None,
                type=click.INT,
                help="Number of epochs to train",
                show_default=True)
@click.option('--logger-filename',
                default='logger.log',
                type=click.Path(
                    file_okay=True,
                    dir_okay=False,
                    writable=True),
                help="Filename of logger",
                show_default=True)
@click.option('--weight-file',
                default=None,
                type=click.Path(
                    file_okay=True,
                    dir_okay=False,
                    writable=True),
                help="Filename to load model weights",
                show_default=True)

def main(batch_size,
        training_dir,
        checkpoint_dir,
        epochs,
        n_fixed_layers,
        logger_filename,
        weight_file):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    df_train = pd.read_csv("dataset/training.csv")
    df_train = df_train.sample(frac=1,random_state=42)
    df_val = pd.read_csv("dataset/validation.csv")

    #3 using subset of training data
    df_train['Filename'] = training_dir+"/"+df_train['Filename'].astype(str)
    df_val['Filename'] = training_dir+"/"+df_val['Filename'].astype(str)
    #df_train = df_train[:100]
    #df_val = df_val[:100]
    generator = preprocess.tfdata_generator(df_train['Filename'].values,
                                        df_train['Drscore'].values,
                                        is_training=True,
                                        buffer_size=50,
                                        batch_size=batch_size)

    validation_generator = preprocess.tfdata_generator(df_val['Filename'].values,
                                        df_val['Drscore'].values,
                                        is_training=False,
                                        buffer_size=50,
                                        batch_size=batch_size)

    ## various callbacks
    tensorboard_cbk = callbacks.TensorBoard(log_dir=checkpoint_dir,
                                                    update_freq='epoch',
                                                    write_grads=False,
                                                    histogram_freq=0)
    checkpoint_cbk = callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir,'weights-{epoch:03d}.hdf5'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False)

    earlystop_ckb = callbacks.EarlyStopping(monitor='val_loss',
                        patience=5,
                        restore_best_weights=False)
    csv_callback = callbacks.CSVLogger(os.path.join(checkpoint_dir,logger_filename),append=True)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6)
    lr_scheduler = callbacks.LearningRateScheduler(step_decay)

    model,base_model = create_model()

    ## freeze upper layers
    files = sorted(glob(os.path.join(checkpoint_dir, 'weights-*.hdf5')))
    if weight_file:
        model_file = weight_file
        initial_epoch = int(model_file[-8:-5])
        print('Resuming using saved model %s.' % model_file)
        model = tf.keras.models.load_model(model_file)
    elif files:
        model_file = files[-1]
        initial_epoch = int(model_file[-8:-5])
        print('Resuming using saved model %s.' % model_file)
        model = tf.keras.models.load_model(model_file)
    else:
        #model,base_model = create_model()
        initial_epoch = 0

    if n_fixed_layers:
        for layer in base_model.layers[:n_fixed_layers]:
            layer.trainable = False
        for layer in base_model.layers[n_fixed_layers:]:
            layer.trainable = True
            print("training layer {}".format(layer.name))

    model.fit(
        generator,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=df_train.shape[0]//batch_size,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=df_val.shape[0]//batch_size,
        callbacks=[tensorboard_cbk,
                    checkpoint_cbk,
                    csv_callback,
                    reduce_lr])

if __name__ == "__main__":
    exit(main())  # pragma: no cover
