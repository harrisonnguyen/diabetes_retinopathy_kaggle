#!/usr/bin/env bash
python evaluate.py --batch-size 16 --weight-file "/home/harrison/tensorflow_checkpoints/diabetes/inceptionv3_mse/weights-050.hdf5" --output-file "testSubmissionMSE.csv"
