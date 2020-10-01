This repo is forked from https://github.com/tkwoo/anogan-keras, and modified to deal with credit card fraud detection. 

The dataset is taken from https://www.kaggle.com/mlg-ulb/creditcardfraud. 

How to Run:

#train & test
python main_credit.py --mode train --epoch xx (--label_idx 0/1  --img_idx yy)
#test only
python main_credit.py (--label_idx 0/1  --img_idx yy)
