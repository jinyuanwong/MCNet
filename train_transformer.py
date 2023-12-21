
"""df_metrics.csv 里面sensitivity计算可能有问题注意修改"""
"""
This is used to stage low+mid(8-23) HAMD and High(>=24) HAMD score MDD subjects with label 0 and 1 

Normalization Method: Layer Normalization (Single Sample Normalization)

Data Augmentation: None

"""


# 使用当前时间作为随机种子
from wandb.keras import WandbCallback
import sys
import time
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import *
import tensorflow as tf
import tensorflow.keras as keras
from datetime import date
import numpy as np
import random
import tensorflow_addons as tfa
import wandb
import config
current_time = int(time.time())

# set the random seed
random.seed(current_time)
np.random.seed(current_time)
tf.random.set_seed(current_time)

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*6)])
# 保存日志

preprocessed_hb_fold_path = config.PREPROCESSED_HB_FOLD_PATH
default_hb_fold_path = config.DEFAULT_HB_FOLD_PATH

# hbo_fold_path = './allData/Output_npy/twoDoctor/nor-all-hbo-hc-mdd'
num_of_k_fold = 10
# /home/jy/Documents/JinyuanWang_pythonCode/results/wang_alex/HbO-All-HC-MDD


def split_k_fold_cross_validation(data, label, k, num_of_k_fold, adj=None):
    total_number = data.shape[0]
    one_fold_number = total_number//num_of_k_fold
    X_val = data[k*one_fold_number:(k+1)*one_fold_number]
    Y_val = label[k*one_fold_number:(k+1)*one_fold_number]
    X_train = np.concatenate(
        (data[0:k*one_fold_number], data[(k+1)*one_fold_number:]))
    Y_train = np.concatenate(
        (label[0:k*one_fold_number], label[(k+1)*one_fold_number:]))

    if adj is None:
        return X_train, Y_train, X_val, Y_val
    else:
        adj_val = adj[k*one_fold_number:(k+1)*one_fold_number]
        adj_train = np.concatenate(
            (adj[0:k*one_fold_number], adj[(k+1)*one_fold_number:]))
        return X_train, Y_train, X_val, Y_val, adj_train, adj_val


class TrainModel():
    def __init__(self, model_name, sweep_config=None):
        self.batch_size = 256
        self.epochs = config.MAX_EPOCHS
        # ['nor-all-hbo-hc-mdd']  # 'HbO-All-Three'
        self.all_archive = config.INPUT_HB_TYPE
        self.all_classifiers = [model_name]
        self.repeat_count_all = 1
        self.sweep_config = sweep_config
        self.parameter = config.PARAMETER[model_name]
        self.hb_path = self.parameter.get('hb_path')
    def begin(self):
        
        epochs = self.epochs

        for archive in self.all_archive:
            hbo_fold_path = default_hb_fold_path + archive
            fnirs_data_path = preprocessed_hb_fold_path + \
                model_name if model_name in config.MODELS_NEED_PREPROCESS_DATA else hbo_fold_path
            for classifier_name in self.all_classifiers:

                # Read data and split into training+validation and testing set with a ratio of 9:1
                # case - not using adj
                # case using adj include GNN, GNN-Transformer, ....

                X_train_val, X_test, Y_train_val, Y_test = read_data_fnirs_mcnet(
                    fnirs_data_path, model_name, self.hb_path, None)
                for k in range(num_of_k_fold):

                    X_train, Y_train, X_val, Y_val = split_k_fold_cross_validation(
                        X_train_val, Y_train_val, k, num_of_k_fold)

                    output_directory = os.getcwd() + '/results/' + classifier_name + '/' + \
                        archive + \
                        '/' + 'k-fold-' + str(k) + '/'
                    create_directory(output_directory)

                    checkpoint_path = output_directory + '/checkpoint'

                    def learning_rate_schedule(epoch, learning_rate):
                        return learning_rate

                    lr_monitor = tf.keras.callbacks.LearningRateScheduler(
                        learning_rate_schedule)


                    input_shape = [self.batch_size,
                                    X_train_val.shape[1],
                                    X_train_val.shape[2],
                                    1]
                    
                    for repeat_count in range(self.repeat_count_all):

                        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                                           monitor= 'val_' + config.MONITOR_METRIC,
                                                           mode = 'max',
                                                           save_weights_only=True,
                                                           save_best_only=True)

                        callbacks = [model_checkpoint,
                                     lr_monitor]

                        tf.keras.backend.clear_session()
                        print(
                            f'Current / Total repeat count: {repeat_count} / {self.repeat_count_all}')

                        model = self.create_classifier(
                            classifier_name, output_directory, callbacks, input_shape, epochs, self.sweep_config)

                        model.fit(X_train, Y_train, X_val,
                                      Y_val, X_test, Y_test)

                        del model

                        

    def create_classifier(self, classifier_name, output_directory, callbacks, input_shape, epochs, sweep_config=None):

        if classifier_name == 'comb_cnn':  # Time-CNN
            from classifiers import comb_cnn
            return comb_cnn.Classifier_CNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        else:
            raise Exception('Your error message here')

if __name__ == '__main__':
    arg = sys.argv
    model_name = arg[1]
    do_individual_normalize = True
    info = {'current_time_seed': current_time,
            'message': arg[2],
            'parameter': config.PARAMETER[model_name],
            'monitor_metric': config.MONITOR_METRIC
            }
    print('You are using model: {}'.format(model_name))

    model = TrainModel(model_name)
    model.begin()
