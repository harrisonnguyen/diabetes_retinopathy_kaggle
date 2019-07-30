import tensorflow as tf
import pandas as pd
import preprocess
from glob import glob
from model import create_model
from metrics import quadratic_weighted_kappa,sklearn_quadratic_weighted_kappa
import numpy as np
import os
import click

def change_file_ext(x):
    if 'tiff' == x.split(".")[-1] or 'tif' == x.split(".")[-1]:
        x = x.split(".")[0] + '.jpeg'
    return x

def predict_test(model,test_dir,batch_size,output_file):

    test_df = pd.read_csv("dataset/SampleSubmission.csv")
    test_df['Id'] = test_df['Id'].apply(change_file_ext)
    test_df['Id'] = test_dir +"/" +test_df['Id'].astype(str)

    test_generator = preprocess.tfdata_generator(test_df['Id'].values,
                                        is_training=False,
                                        batch_size=batch_size,
                                        n_epochs=1)
    predictions = model.predict(test_generator,steps=test_df.shape[0]//batch_size+1)
    df = pd.read_csv("dataset/SampleSubmission.csv")
    classes = np.argmax(predictions,axis=1)
    df["Expected"] = np.argmax(predictions,axis=1)
    df.to_csv(output_file,index=False)

    return df

def evaluate_validation(model,training_dir,batch_size):
    df_val = pd.read_csv("dataset/validation.csv")
    df_val['Filename'] = training_dir+"/"+df_val['Filename'].astype(str)
    validation_generator = preprocess.tfdata_generator(df_val['Filename'].values,
                                        df_val['Drscore'].values,
                                        is_training=False,
                                        batch_size=batch_size,
                                        n_epochs=1)
    validation_prediction = model.predict(validation_generator,
                                        steps=df_val.shape[0]//batch_size+1)
    kappa = sklearn_quadratic_weighted_kappa(
                np.argmax(validation_prediction,axis=1),
                df_val['Drscore'].values)
    return kappa
@click.command()
@click.option('--weight-file',
                default=None,
                type=click.Path(
                    file_okay=True,
                    dir_okay=False,
                    writable=True),
                help="Filename to load model weights",
                show_default=True)
@click.option('--checkpoint-dir',
                default='/home/harrison/tensorflow_checkpoints/diabetes/',
                type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
                help="dir checkpoints",
                show_default=True)
@click.option('--batch-size',
                default=8,
                type=click.INT,
                help="Batch size",
                show_default=True)
@click.option('--test-dir',
            default='dataset/Test_jpeg',
                type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
                help="dir for test data",
                show_default=True)
@click.option('--training-dir',
                default='dataset/output_combined2',
                type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
                help="dir checkpoints",
                show_default=True)
@click.option('--output-file',
                default='testSubmission.csv',
                type=click.Path(
                    file_okay=True,
                    dir_okay=False,
                    writable=True),
                help="name of file to write test results",
                show_default=True)
def main(weight_file,checkpoint_dir,
        batch_size,test_dir,test,
        training_dir,output_file):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

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

    print(evaluate_validation(model,training_dir,batch_size))
    predict_test(model,test_dir,batch_size,output_file)
    #print(evaluate_validation(model,training_dir,batch_size))

if __name__ == "__main__":
    exit(main())  # pragma: no cover
