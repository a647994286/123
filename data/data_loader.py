import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.spacefeatures import space_features

import warnings

warnings.filterwarnings('ignore')





class Dataset_BJ13(Dataset):
    """
        Dataset_BJ13:
        A PyTorch Dataset class for loading and preprocessing the BJ13 urban traffic dataset.
        Includes spatial and temporal embeddings, data normalization, and window slicing for sequence modeling.
        """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Dataset_BJ13',
                 target='OT', scale=True, inverse=False, timeenc=1, freq='h', cols=None):
        """
            Initialize the dataset instance.

            Args:
                root_path: root directory of the dataset
                flag: one of ['train', 'val', 'test'] indicating data split
                size: list [seq_len, label_len, pred_len], defines window structure
                features: unused in current version, kept for compatibility
                data_path: CSV file name or relative path
                target: target variable name (e.g., 'OT')
                scale: whether to apply normalization
                inverse: whether to return unscaled ground truth
                timeenc: if 1, encode time features; else use raw
                freq: time granularity, e.g., 'h' for hourly
                cols: list of columns to be used (optional)
            """
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """
            Load and preprocess the BJ13 dataset:
            - Read raw CSV data
            - Select target & feature columns
            - Split into train/val/test sets
            - Apply normalization (StandardScaler)
            - Extract time and space embeddings
            """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove('time')
        df_raw = df_raw[['time'] + cols]

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        location = df_raw.columns[1:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "BJ13"))
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

    def __getitem__(self, index):
        """
            Retrieve one sample window from the dataset.

            Returns:
                seq_x: input sequence [seq_len, num_features]
                seq_y: target sequence [label_len + pred_len, num_features]
                seq_x_mark: temporal features [seq_len, time_dim]
                space_mark_x: spatial features [seq_len, num_nodes, spatial_dim]
            """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_time_stamp[s_begin:s_end]
        space_mark_x = np.tile(self.data_space_stamp, (self.seq_len, 1, 1))
        return seq_x, seq_y, seq_x_mark, space_mark_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

"""
    Dataset_Bike2:
    Similar to Dataset_BJ13 but for the second version of BikeNYC (BikeNYC2).
    Implements same sliding window mechanism and embedding structures.
    Differences:
         - Train/val/test split: 60% / 20% / 20%
         - Dataset name: 'Dataset_Bike2'
    """
class Dataset_Bike(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Dataset_Bike.csv',
                 target='OT', scale=True, inverse=False, timeenc=1, freq='h', cols=None):
        # 滑窗参数 [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 12
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.win_len = self.seq_len + self.pred_len

        # 数据划分标识
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 获取特征列
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove('time')
        df_raw = df_raw[['time'] + cols]
        df_data = df_raw[cols]

        if self.scale:
            self.scaler.fit(df_data.values[:2942])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data
        self.data_y = data if not self.inverse else df_data.values

        df_stamp = df_raw[['time']]
        df_stamp['time'] = pd.to_datetime(df_stamp['time'])
        self.data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        location = df_data.columns.values
        data_space_stamp = []
        for loc in location:
            loc_vector = [float(val) for val in loc.split(',')]
            data_space_stamp.append(loc_vector)
        self.data_space_stamp = np.array(space_features(data_space_stamp, "Bike"))

        total_len = len(self.data)
        self.total_samples = total_len - self.seq_len - self.pred_len + 1
        all_indices = list(range(self.total_samples))

        train_num, val_num, test_num = 2621, 874, 874
        assert train_num + val_num + test_num <= self.total_samples

        if self.set_type == 0:
            self.indices = all_indices[:train_num]
        elif self.set_type == 1:
            self.indices = all_indices[train_num:train_num + val_num]
        else:
            self.indices = all_indices[train_num + val_num:train_num + val_num + test_num]

    def __getitem__(self, idx):
        index = self.indices[idx]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = (
            np.concatenate(
                [self.data[s_begin:s_begin + self.label_len],
                 self.data_y[s_begin + self.label_len:r_end]], 0)
            if self.inverse else self.data_y[r_begin:r_end]
        )
        seq_x_mark = self.data_time_stamp[s_begin:s_end]
        space_mark_x = np.tile(self.data_space_stamp, (self.seq_len, 1, 1))

        return seq_x, seq_y, seq_x_mark, space_mark_x

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Bike2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Dataset_Bike2',
                 target='OT', scale=True, inverse=False, timeenc=1, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove('time')
        df_raw = df_raw[['time'] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        location = df_raw.columns[1:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "Bike"))
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_time_stamp[s_begin:s_end]
        space_mark_x = np.tile(self.data_space_stamp, (self.seq_len, 1, 1))
        return seq_x, seq_y, seq_x_mark, space_mark_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TaxiNYC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Dataset_TaxiNYC',
                 target='OT', scale=True, inverse=False, timeenc=1, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove('time')
        df_raw = df_raw[['time'] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        location = df_raw.columns[1:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "TaxiNYC"))
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_time_stamp[s_begin:s_end]
        space_mark_x = np.tile(self.data_space_stamp, (self.seq_len, 1, 1))
        return seq_x, seq_y, seq_x_mark, space_mark_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)