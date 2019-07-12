import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from tensorflow.keras.applications.xception import Xception

from sklearn.model_selection import train_test_split
from metrics import quadratic_weighted_kappa
# create the base pre-trained model
import preprocess
import click

def change_file_ext(x):
    if 'tiff' == x.split(".")[-1] or 'tif' == x.split(".")[-1]:
        x = x.split(".")[0] + '.jpeg'
    return x
@click.command

def main(batch_size):
    df = pd.read_csv("dataset/training-labels.csv")
    df['Filename'] = df['Filename'].apply(change_file_ext)
    train_ind,val_ind = train_test_split(range(df.shape[0]),test_size=0.2,random_state=42)

    df['Filename'] = 'dataset/output_combined2/' + df['Filename'].astype(str)

    generator = preprocess.tfdata_generator(df['Filename'][train_ind].values,
                                        df['Drscore'][val_ind].values,
                                        is_training=True,
                                        buffer_size=len(train_ind),
                                        batch_size=batch_size)

    validation_generator = preprocess.tfdata_generator(df['Filename'][val_ind].values,
                                        df['Drscore'][val_ind].values,
                                        is_training=False,
                                        buffer_size=len(val_ind),
                                        batch_size=batch_size)

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

    ## freeze upper layers
    for layer in base_model.layers:
        layer.trainable = False

    ## various callbacks
    tensorboard_cbk = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                    update_freq='epoch',
                                                    write_grads=True,
                                                    histogram_freq=1)
    checkpoint_cbk = keras.callbacks.ModelCheckpoint(
        filepath='weights.{epoch:02d}.hdf5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        load_weights_on_restart=False)
    earlystop_ckb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                        patience=5,
                        restore_best_weights=False)


    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
                metrics=[lambda x,y:quadratic_weighted_kappa(x,y,5),
                        'accuracy'])

    model.fit(
        generator,
        epochs=5,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[tensorboard_cbk,
                    checkpoint_cbk])
    test_df = pd.read_csv("dataset/SampleSubmission.csv")
    test_df['Filename'] = test_df['Filename'].apply(change_file_ext)
    test_df['Filename'] = 'dataset/Test_jpeg/' + test_df['Filename'].astype(str)

    test_generator = preprocess.tfdata_generator(test_df['Filename'].values,
                                        test_df['Drscore'].values,
                                        is_training=False)
