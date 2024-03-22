import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear,HANConv
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from globel_args import device
# HGTConv = HANConv
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from utils import get_metrics




class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            #  in_channels: Union[int, Dict[str, int]],
            conv = HGTConv(-1, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.fc = Linear(hidden_channels*2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.pkl_ctl = None
        self.best_auc = 0.0
        self.param = None

    def forward(self,data, edge_index):
        x_dict_, edge_index_dict = data['x_dict'], data['edge_dict']
        x_dict = x_dict_.copy()#创建副本
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        all_list = []
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            all_list.append(x_dict.copy())

        for i,_ in x_dict_.items():
            x_dict[i] =torch.cat(tuple(x[i] for x in all_list), dim=1)

        m_index = edge_index[0]
        d_index = edge_index[1]
        self.save_data = x_dict
        self.edge_index = edge_index
        Em = self.dropout(x_dict['n1'])
        Ed = self.dropout(x_dict['n2'])

        y = Em@Ed.t()
        # y = torch.cat((Em, Ed), dim=1)
        # y = self.fc(y)
        y = y[m_index,d_index].unsqueeze(-1)
        return y




    def save_model_state(self, kf, train_idx, test_idx,y):
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.y = y
        self.concat_same_m_d(kf)


    def concat_same_m_d(self,kf):
        data_concat = torch.concat((self.save_data['n1'][self.edge_index[0]],self.save_data['n2'][self.edge_index[1]]), dim=1).cpu().numpy()
        train_data_concat = data_concat[self.train_idx]
        test_data_concat = data_concat[self.test_idx]
        # print(train_data_concat.shape)

        joblib.dump({'train_data': train_data_concat,
                     'test_data':test_data_concat,
                     'y_train': self.y[self.train_idx].cpu().numpy(),
                     'y_test': self.y[self.test_idx].cpu().numpy(),
                     'all_data':{'Em':self.save_data['n1'].cpu().numpy(),
                                 'Ed':self.save_data['n2'].cpu().numpy()},
                     },
                    './mid_data/' + str(6) + 'nl' + str(kf) + 'kf_best_cat_data.dict')


class ModelSelector:
    def __init__(self):
        self.models = {
            'xgboost': XGBClassifier(),
        }
        self.param_grids = {

            'xgboost': {'max_depth': [6], 'learning_rate': [ 0.15]},

        }
    def get_models(self, model_list=[]):
        if model_list == []:
            return self.models
        else:
            models_dict = {}
            for key in model_list:
                models_dict[key] = self.models[key]
            return models_dict

    # 直接验证
    def train_with_grid_search(self, X_train, y_train, X_test, y_test, models_dict={}):
        if models_dict=={}:
            models_dict = self.models
        ls_dict = {}
        for model_name, model in models_dict.items():
            ls_dict = []
            param_grid = self.param_grids[model_name]
            grid = ParameterGrid(param_grid)
            best_score = -1
            best_params = None
            for params in grid:
                print(params)
                model.set_params(**params)
                model.fit(X_train, y_train)
                y_score = model.predict_proba(X_test)
                auc_all = get_metrics(y_test, y_score[:,1])
                auc = auc_all[0][2]

                ls_dict.append([auc_all[0]])

                if auc > best_score:
                    best_score = auc
                    best_params = params
            print(f"Best parameters for {model_name}: {best_params}, Best auc score: {best_score}")

        return ls_dict

