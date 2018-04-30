#!/bin/bash
wget 'https://www.dropbox.com/s/00ixy3d79ktt0mr/fcn8s_model.h5?dl=1' -O fcn8s_model.h5
python3 Predict.py -m './fcn8s_model.h5' -i $1 -o $2