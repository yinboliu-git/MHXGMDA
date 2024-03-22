import multiprocessing

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
import os
from utils import get_data
from train_model import CV_train

import time

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
import joblib
from models import ModelSelector,HGT


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    def __init__(self):
        self.datapath = './data/'
        self.save_file = 'save_file/'

        self.kfold = 5
        self.maskMDI = False
        self.hidden_channels = 512  # 256 512
        self.num_heads = 4  # 4 8
        self.num_layers = 4  # 4 8
        self.self_encode_len = 256
        self.globel_random = 222  # 120
        self.other_args = {'arg_name': [], 'arg_value': []}

        # 解码参数
        self.epochs = 2000  ## 1000
        self.print_epoch = 20 ## 20


def set_attr(config, param_search):
    param_grid = param_search
    param_keys = param_grid.keys()
    param_grid_list = list(ParameterGrid(param_grid))
    for param in param_grid_list:
        config.other_args = {'arg_name': [], 'arg_value': []}
        for keys in param_keys:
            setattr(config, keys, param[keys])
            config.other_args['arg_name'].append(keys)
            print(keys, param[keys])
            config.other_args['arg_value'].append(param[keys])
        yield config#迭代
    return 0


class Data_paths:
    def __init__(self):
        self.paths = './data/'
        self.md = self.paths + 'c_d.csv'
        self.mm = [self.paths + 'c_gs.csv', self.paths + 'c_ss.csv']
        self.dd = [self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']


best_param_search = {
    'hidden_channels': [64, 128, 256, 512],
    'num_heads': [4, 8, 16, 32],
    'num_layers': [2, 4, 6, 8],
    # 'CL_margin' :[0.5,1.0,1.5,2.0],
    'CL_noise_max': [0.05, 0.1, 0.2, 0.4],
}




if __name__ == '__main__':

    set_seed(521)

    best_param_search = {
        'hidden_channels': [64],
        'num_heads': [8],
        'num_layers': [6],
    }
    param_search = best_param_search
    save_file = '5cv_data_1000'
    params_all = Config()
    param_generator = set_attr(params_all, param_search)
    data_list = []
    filepath = Data_paths()

    while True:
        try:
            params = next(param_generator)
        except:
            break

        data, y, edg_index_all = get_data(file_pair=filepath, params=params)

        data_tuple = get_data(file_pair=filepath, params=params)
        data_idx, auc_name = CV_train(params, data_tuple)  # 交叉验证


    for i in range(1, 6):
        kf = i
        file_name = './mid_data/' + str(6) + 'nl' + str(kf) + 'kf_best_cat_data.dict'
        while True:
            if os.path.exists(file_name):
                break
            else:
                time.sleep(1)
                continue

        data_load = joblib.load(file_name)
        print( './mid_data/' + str(kf) + 'kf_best_cat_data.dict')
        selector = ModelSelector()
        X_train, X_test, y_train, y_test = data_load['train_data'], data_load['test_data'], data_load['y_train'], data_load[
            'y_test']

        # 获取模型并进行训练
        model_list = []  # 选择模型
        models = selector.get_models(model_list)

        ls_dict = selector.train_with_grid_search(X_train, np.reshape(y_train, (-1,)), X_test,
                                                  np.reshape(y_test, (-1,)), models)  
        data_list.append(ls_dict)

        if data_list.__len__() > 1:
            data_all = np.concatenate(tuple(x for x in data_list), axis=1)
        else:
            data_all = data_list[0]  # 或者采取其他适当的操作

        if data_all is not None:
            np.save(params_all.save_file + save_file + '.npy', data_all)
    # 其他操作

    data_idx = np.load(params_all.save_file + save_file + '.npy', allow_pickle=True)

    data_mean = data_idx[:, :, 2:].mean(0)
    idx_max = data_mean[:, 1].argmax()
    print()
    print('最大值为：')
    print(data_mean[idx_max, :])
