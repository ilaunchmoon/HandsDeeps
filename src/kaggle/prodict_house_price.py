import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from torch.utils import data
from src.utils.download_tool import DATA_HUB, DATA_URL, download
from src.visualization.plot import plot

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_raw_data():
    DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', 
                                      '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 
                                     'fa19780a7b011d9009e8bff8e99922a8ee2eb90')
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))  
    return train_data, test_data


def get_feature_data(train_data, test_data):
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    return all_features


def preprocess_data(all_feature, train_data):
    # 找到所有数值类型的特征
    numeric_features = all_feature.dtypes[all_feature.dtypes != 'object'].index

    # 对所有数值特征进行标准化
    all_feature[numeric_features] = all_feature[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    
    # 填充 NaN 值
    all_feature[numeric_features] = all_feature[numeric_features].fillna(0)
    
    # 对分类特征进行 One-Hot 编码（dummy_na=True 保留 NaN 值）
    all_feature = pd.get_dummies(all_feature, dummy_na=True)

    # 确保数据是数值类型
    all_feature = all_feature.astype('float32')

    # 分割训练集和测试集
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_feature[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_feature[n_train:].values, dtype=torch.float32)
    
    # 获取目标标签，确保是数值类型
    train_labels = pd.to_numeric(train_data.SalePrice, errors='coerce').fillna(0).values
    train_labels = torch.tensor(train_labels.reshape(-1, 1), dtype=torch.float32)  # 确保标签是 (N, 1)
    
    return train_features, test_features, train_labels



def get_net(in_feature):
    return nn.Sequential(nn.Linear(in_feature, 1))



def get_log_rmse(net, loss, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    
    # 确保 labels 是 Tensor 类型
    if isinstance(labels, pd.DataFrame):  # 如果是 DataFrame 类型，转换为 Tensor
        labels = labels.apply(pd.to_numeric, errors='coerce')  # 转换为数值类型，非数值转换为 NaN
        labels = torch.tensor(labels.values, dtype=torch.float32)
    
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimer.step()
        train_ls.append(get_log_rmse(net, loss, train_features, train_labels)) 
        if test_labels is not None:
            test_ls.append(get_log_rmse(net, loss, test_features, test_labels)) 
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size, net):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                 xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                 legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}',
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(net, train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    train_ls, _ = train(net, train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size)
    plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse: {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./result_data/submission.csv', index=False)




if __name__ == "__main__":
    k, num_epochs, lr, weight_decay, batch_size = 10, 100, 0.05, 0, 64
    train_data, test_data = get_raw_data()
    all_features = get_feature_data(train_data, test_data)
    train_features, test_features, train_labels = preprocess_data(all_features, train_data)
    loss = nn.MSELoss()
    net = get_net(train_features.shape[1])
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, net)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, ', 
          f'平均验证log rmse: {float(valid_l):f}')
    train_and_pred(net, train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)

    



