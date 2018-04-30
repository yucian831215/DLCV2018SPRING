#!/bin/bash
wget 'https://www.dropbox.com/s/twg374onud05l09/fcn32s_model.h5?dl=1' -O fcn32s_model.h5
python3 Predict.py -m './fcn32s_model.h5' -i $1 -o $2