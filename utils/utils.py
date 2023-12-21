from builtins import print
import numpy as np
import pandas as pd
import matplotlib
from scipy.interpolate import interp1d
from scipy.io import loadmat
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator
import tensorflow as tf
import utils


from utils.utils_mine import z_norm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

# def get_specificity(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return tn / (tn+fp)



def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_past_value(directory,name):
    print(f'directory : {directory}')
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files: 
            if file[:3] == 'his':
                location = directory + file
                history = pd.read_csv(location)
                print(f'File Location: {location}')
                return np.max(history['val_accuracy']),np.max(history['val_'+name])
    return 0,0
