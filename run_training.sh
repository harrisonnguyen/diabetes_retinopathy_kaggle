#!/usr/bin/env bash
python model.py --epochs 80 --checkpoint-dir '/home/harrison/tensorflow_checkpoints/diabetes/inceptionv3_weight' --batch-size 12 --class-weight
