import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from tensorflow.keras.applications.xception import Xception

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
def change_file_ext(x):
    if 'tiff' == x.split(".")[-1] or 'tif' == x.split(".")[-1]:
        x = x.split(".")[0] + '.jpeg'
    return x

def create_model():
    base_model = Xception(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

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
@click.option('--test-dir',
            default='dataset/Test_jpeg',
                type=click.Path(
                    file_okay=False,
                    dir_okay=True,
                    writable=True),
                help="dir for test data",
                show_default=True)
@click.option('--checkpoint-dir',
                default='/home/harrison/tensorflow_checkpoints/diabetes',
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

def main(batch_size,
        training_dir,
        test_dir,
        checkpoint_dir,
        epochs):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    df = pd.read_csv("dataset/training-labels.csv")
    df['Filename'] = df['Filename'].apply(change_file_ext)


    #3 using subset of training data
    data_root = pathlib.Path(training_dir)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path).split("/")[-1] for path in data_root.iterdir()]
    df = df[df['Filename'].isin(all_image_paths)]

    df['Filename'] = training_dir+"/"+df['Filename'].astype(str)
    train_ind,val_ind = train_test_split(df.index.values,test_size=0.2,random_state=42)

    generator = preprocess.tfdata_generator(df['Filename'][train_ind].values,
                                        df['Drscore'][train_ind].values,
                                        is_training=True,
                                        buffer_size=len(train_ind),
                                        batch_size=batch_size)

    validation_generator = preprocess.tfdata_generator(df['Filename'][val_ind].values,
                                        df['Drscore'][val_ind].values,
                                        is_training=False,
                                        buffer_size=len(val_ind),
                                        batch_size=batch_size)

    ## various callbacks
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir,
                                                    update_freq='epoch',
                                                    write_grads=True,
                                                    histogram_freq=1)
    checkpoint_cbk = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir,'weights-{epoch:03d}.hdf5'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False)
    earlystop_ckb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                        patience=5,
                        restore_best_weights=False)



    model,base_model = create_model()

    ## freeze upper layers
    files = sorted(glob(os.path.join(checkpoint_dir, 'weights-*.hdf5')))
    if files:
        model_file = files[-1]
        initial_epoch = int(model_file[-8:-5])
        print('Resuming using saved model %s.' % model_file)
        model = tf.keras.models.load_model(model_file)
    else:
        model,base_model = create_model()
        initial_epoch = 0

    for layer in base_model.layers:
        layer.trainable = False
    model.fit(
        generator,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=len(train_ind)//batch_size,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=len(val_ind)//batch_size,
        callbacks=[tensorboard_cbk,
                    checkpoint_cbk])
    iterator = validation_generator.make_one_shot_iterator()
    next_element = iterator.get_next()
    truth = np.array([])
    for i in range(len(val_ind)//batch_size):
        try:
            x,y = sess.run(next_element)
            truth = np.append(truth,y)
        except:
            break
    validation_prediction = model.predict(validation_generator,steps=len(val_ind)//batch_size)
    print(sklearn_quadratic_weighted_kappa(np.argmax(validation_prediction,axis=1),truth))

    """
    test_df = pd.read_csv("dataset/SampleSubmission.csv")
    test_df['Id'] = test_df['Id'].apply(change_file_ext)
    test_df['Id'] = test_dir +"/" +test_df['Id'].astype(str)

    test_generator = preprocess.tfdata_generator(test_df['Id'].values,
                                        is_training=False)
    predictions = model.predict(test_generator,steps=len(test_df['Id'].values)//batch_size)
    print(np.argmax(predictions,axis=1))
    """
if __name__ == "__main__":
    exit(main())  # pragma: no cover
