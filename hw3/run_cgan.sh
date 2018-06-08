#!/bin/bash 
wget -O cgan_model.data-00000-of-00001 https://www.dropbox.com/s/e5vihgqjqnxpt88/model-60.data-00000-of-00001?dl=1
wget -O cgan_model.index https://www.dropbox.com/s/fraviedljmfjkvg/model-60.index?dl=1
wget -O cgan_model.meta https://www.dropbox.com/s/ewisd30u7jghuz9/model-60.meta?dl=1
python CGAN.py $1