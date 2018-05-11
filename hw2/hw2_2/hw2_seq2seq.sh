#!/bin/bash
wget https://github.com/b02901156/MLDS_HW2/releases/download/ver1/hw2_model.zip
unzip hw2_model.zip
python3 predict.py $1 $2