# MCNet v 0.1.0

## Description
This repo provides implementation of the code for the paper "Interpretable deep learning model for major depressive disorder assessment based on functional near-infrared spectroscopy".

![MCNet](figures/MCNet.png)

## Packages/Dependencies
- `scikit-learn==0.24.1`
- `matplotlib==3.3.4`
- `QtPy==1.9.0`
- `jupyter==1.0.0`
- `Keras==2.3.1`
- `numpy==1.20.1`
- `pandas==1.2.4`
- `tensorflow[and-cuda]==2.12.0`
- `wandb==0.15.11`
- `tensorflow-addons==0.20.0`

## Input data 

You should modify and insert your input data, and modify the condig.py file accordingly.

For example, 
- data(shape: 100, 52, 125) is stored in './allData/Output_npy/twoDoctor/HbO-All-HC-MDD/data.npy'
- label(shape: 100,) in stored in './allData/Output_npy/twoDoctor/HbO-All-HC-MDD/label.npy' 

In config.py
```
DEFAULT_HB_FOLD_PATH = './allData/Output_npy/twoDoctor/'
```
## How to use

```
python train.py comb_cnn
```


### View the result

The output will be stored in './results/comb_cnn/HbO-All-HC-MDD'

