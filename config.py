import numpy as np
# some important parameters

# INPUT_HB_TYPE = ['HbO-All-Lowmid-High'] # 

# INPUT_HB_TYPE = ['HbO-right-VPC-classification',
#                 'HbO-right-STG-classification',
#                 'HbO-MPC-classification']
        # 'HbO-left-DPC-classification',
        # 'HbO-right-DPC-classification',
        # 'HbO-left-PSFC-classification',
#         'HbO-right-PSFC-classification'
#         ]

INPUT_HB_TYPE = ['HbO-All-HC-MDD']

# INPUT_HB_TYPE = [
#                 'HbO-left-STG-classification',
#                 'HbO-right-STG-classification',
#                 'HbO-left-VPC-classification',
#                 'HbO-right-VPC-classification',
#                 'HbO-MPC-classification',
#                 ]

IS_USING_WANDB = True

MAX_EPOCHS = 1000

MODELS_NEED_ADJ_MATRIX = ['graphsage_transformer', 
                          'mvg_transformer', 
                          'gnn_transformer', 
                          'yu_gnn', 
                          'gnn', 
                          'mgn_transformer', 
                          'mgm_transformer']

MODELS_NEED_PREPROCESS_DATA = ['chao_cfnn', 'wang_alex', 'zhu_xgboost', 'yu_gnn'] # left_to_do SVM_ZHIFEI, RSFC_DUAN, NMF_ZHONG

PREPROCESSED_HB_FOLD_PATH = './allData/data_for_reproducing_model/HbO-All-Lowmid-High/'
# PREPROCESSED_HB_FOLD_PATH = './allData/data_for_reproducing_model/HbO-All-HC-MDD/'

DEFAULT_HB_FOLD_PATH = './allData/Output_npy/twoDoctor/'

MONITOR_METRIC = 'accuracy' # 'val_accuracy' 

PARAMETER = {
    'comb_cnn': {
        'hb_path': 'correct_channel_data.npy',
        }
} 
