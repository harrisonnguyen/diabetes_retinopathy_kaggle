import tensorflow as tf
import preprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pathlib

def main():
    training_dir = 'dataset/output_combined2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    df = pd.read_csv("dataset/training-labels.csv")

    """
    data_root = pathlib.Path(training_dir)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path).split("/")[-1] for path in data_root.iterdir()]
    df = df[df['Filename'].isin(all_image_paths)]


    batch_size = 1
    train_ind,val_ind = train_test_split(df.index.values,test_size=0.2,random_state=42)
    train_ind.sort()
    val_ind.sort()
    df.loc[train_ind].to_csv("dataset/training.csv",index=False)
    df.loc[val_ind].to_csv("dataset/validation.csv",index=False)
    """
    df_train = pd.read_csv("dataset/training.csv")
    df_val = pd.read_csv("dataset/validation.csv")
    df_train['Filename'] = training_dir+"/"+df_train['Filename'].astype(str)
    df_val['Filename'] = training_dir+"/"+df_val['Filename'].astype(str)
    generator = preprocess.tfdata_generator(
                    df_train['Filename'],
                    df_train['Drscore'],
                    is_training=True,
                    buffer_size=500,
                    batch_size=1)

    validation_generator = preprocess.tfdata_generator(
                            df_val['Filename'].values,
                            df_val['Drscore'].values,
                            is_training=False,
                            buffer_size=500,
                            batch_size=1)
    iterator = generator.make_one_shot_iterator()
    next_element = iterator.get_next()
    i=0
    while True:
        try:
            x,y,file_name = sess.run(next_element)
            print(file_name)
            print(i)
            i+=1
        except:
            break

    i = 0
    iterator = validation_generator.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
        try:
            x,y,file_name = sess.run(next_element)
            print(file_name)
            print(i)
            i+=1
        except:
            break
if __name__ == "__main__":
    exit(main())  # pragma: no cover
