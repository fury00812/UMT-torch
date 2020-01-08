# UMT-torch
An implementation of Unsupervised Machine Translation by pytorch

## requirements
- Python 3.7  
  - Pytorch 1.3+
   
If you use pipenv, you can easily set up the environment as follows.  
```
pipenv install --dev
```

## Unsupervised NMT tutorial
Simple tutorial using data prepared in `data/test/asp_csj`
### 1. Preprocess data
```
cd NMT
./get_data.sh
```
### 2. Tran the NMT model
```
Python main.py
```
