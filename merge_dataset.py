import tensorflow as tf
import preprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pathlib
import shutil
import os

def main():
    training_dir = '/media/harrison/ShortTerm/Users/HarrisonG/Train/new_images'

    data_root = pathlib.Path(training_dir)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path).split("/")[-1] for path in data_root.iterdir()]

    df = pd.read_csv("dataset/training_combined.csv")
    df_true = pd.read_csv("dataset/training-labels.csv")
    for ele in all_image_paths:
        if ele not in df["Filename"].values:

            #df['A'].str.contains("hello")
            #dr = df_true.loc[df_true['Filename'] == ele]
            print(os.path.splitext(ele)[0])
            dr = df_true[df_true['Filename'] == ele]
            if dr.shape[0] == 0:
                dr = df_true[df_true['Filename'] == os.path.splitext(ele)[0]+".tif"]
                filename = dr["Filename"].iloc[0]
                dr["Filename"].iloc[0] = os.path.splitext(ele)[0]+".jpeg"
            #if ".tif" not in ele or ".tiff" not in ele:
            #    os.rename(os.path.join('/media/harrison/ShortTerm/Users/HarrisonG/Train/output_combined2',ele),os.path.join('/media/harrison/ShortTerm/Users/HarrisonG/Train/new_images',ele))
            df = df.append(dr)
    df.to_csv("dataset/training_new.csv",index=False)


if __name__ == "__main__":
    exit(main())  # pragma: no cover
