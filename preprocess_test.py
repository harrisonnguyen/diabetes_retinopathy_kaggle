import tensorflow as tf
import preprocess
import pandas as pd
import numpy as np
import click
import model

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    df = pd.read_csv("dataset/training-labels.csv")
    
    data_root = pathlib.Path(training_dir)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path).split("/")[-1] for path in data_root.iterdir()]
    df = df[df['Filename'].isin(all_image_paths)]

    df['Filename'] = training_dir+"/"+df['Filename'].astype(str)
    print(df)
    print(df.shape)
    train_ind,val_ind = train_test_split(df.index.values,test_size=0.2,random_state=42)

    generator = preprocess.tfdata_generator(
                    df['Filename'][train_ind].values,
                    df['Dscore'][train_ind].values,
                    is_training=True,
                    buffer_size=1000,
                    batch_size=batch_size)

    validation_generator = preprocess.tfdata_generator(df['Filename'][val_ind].values,
                                        df['Drscore'][val_ind].values,
                                        is_training=False,
                                        buffer_size=500,
                                        batch_size=batch_size)
    iterator = generator.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
        try:
            x,y = sess.run(next_element)
        except:
            break

    iterator = validation_generator.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
        try:
            x,y = sess.run(next_element)
        except:
            break            
