import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GCN2Conv, SAGEConv, GATConv, HGTConv, Linear
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from models import HGT,ModelSelector
from globel_args import device
from utils import get_metrics
from sklearn.model_selection import KFold
import torch
import copy
import torch.nn as nn
import torch.optim as optim
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import joblib



def train_model(data,y, edg_index_all, train_idx, test_idx, param, k_number):#data图数据 y标签数据 edg_index_all边索引 train_idx训练数据索引 test_idx测试数据索引 param参数k_number折
    hidden_channels, num_heads, num_layers = (
        param.hidden_channels, param.num_heads, param.num_layers,
    )

    epoch_param = param.epochs

    # 模型构建


    model = HGT(hidden_channels, num_heads=num_heads, num_layers=num_layers, data=data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)#创建一个Adam优化器实例，用于更新模型的参数。学习率设置为0.001，权重衰减设置为0.0002
    data_temp = copy.deepcopy(data)#输入的图数据进行深拷贝，并将结果存储在data_temp变量中



    # 训练模型
    auc_list = []
    model.train()#设置模型训练模式
    model.param = param
    for epoch in range(1, epoch_param+1):
        optimizer.zero_grad()
        model.pkl_ctl = 'train'
        y_train = y[train_idx].to('cpu').detach().numpy()
        out = model(data_temp,
                    edge_index=edg_index_all.to(device))
        # 使用train数据进行训练
        loss = F.binary_cross_entropy_with_logits(out[train_idx].to(device), y[train_idx].to(device))#计算二分类交叉熵损失
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if epoch % param.print_epoch == 0:
            model.pkl_ctl='test'#测试
            print()
            # 模型验证
            model.eval()
            with torch.no_grad():
                # 获得所有数据
                out = model(data_temp,
                            edge_index=edg_index_all)
                # 提取验证集数据
                out_pred_s = out[test_idx].to('cpu').detach().numpy()
                out_pred = out_pred_s
                y_true = y[test_idx].to('cpu').detach().numpy()
                # 计算AUC
                auc = roc_auc_score(y_true, out_pred)

                idx = np.arange(y.shape[0])
                if model.best_auc < auc:
                        model.best_auc = auc
                        model.save_model_state(k_number, train_idx, test_idx,y)
                # 计算所有评价指标
                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)
            auc_list.append(auc_idx)
            model.train()
    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name


def CV_train(param, args_tuple=()):
    data, y, edg_index_all = args_tuple
    idx = np.arange(y.shape[0])
    k_number = 1
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)

    kf_auc_list = []
    for train_idx,test_idx  in kf.split(idx):
        print(f'正在运行第{k_number}折, 共{k_fold}折...')
        auc_idx, auc_name = train_model(data, y, edg_index_all, train_idx, test_idx, param, k_number)

        k_number += 1

        kf_auc_list.append(auc_idx)

    data_idx = np.array(kf_auc_list)

    return data_idx, auc_name