import joblib
import numpy as np
import pickle as pkl
import sys
import os
from .utils import Scaler_NYC, Scaler_Chi


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# high frequency time
high_fre_hour = [6, 7, 8, 15, 16, 17, 18]


def split_and_norm_data(all_data,
                        all_data_small,
                        train_rate=0.6,
                        valid_rate=0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    num_of_time, channel, _, _ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate + valid_rate))
    for index, (start, end) in enumerate(((0, train_line), (train_line, valid_line), (valid_line, num_of_time))):
        if index == 0:
            if channel == 48:  # NYC
                scaler = Scaler_NYC(all_data[start:end, :, :, :])
            if channel == 41:  # Chicago
                scaler = Scaler_Chi(all_data[start:end, :, :, :])
        norm_data = scaler.transform(all_data[start:end, :, :, :])
        X, Y = [], []
        high_X, high_Y = [], []
        for i in range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period
            label = norm_data[t:t + pre_len, 0, :, :]
            period_list = []
            for week in range(week_prior):
                period_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature = norm_data[period_list, :, :, :]
            X.append(feature)
            Y.append(label)
            # NYC/Chicago hour_of_day feature index is [1:25]
            if list(norm_data[t, 1:25, 0, 0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
        yield np.array(X), np.array(Y), np.array(high_X), np.array(high_Y), scaler


def normal_and_generate_dataset(
        all_data_filename,
        all_data_filename_small,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    """
    
    Arguments:
        all_data_filename {str} -- all data filename
    
    Keyword Arguments:
        train_rate {float} -- train rate (default: {0.6})
        valid_rate {float} -- valid rate (default: {0.2})
        recent_prior {int} -- the length of recent time (default: {3})
        week_prior {int} -- the length of week  (default: {4})
        one_day_period {int} -- the number of time interval in one day (default: {24})
        days_of_week {int} -- a week has 7 days (default: {7})
        pre_len {int} -- the length of prediction time interval(default: {1})

    Yields:
        {np.array} -- 
                      X shape：(num_of_sample,seq_len,D,W,H)
                      Y shape：(num_of_sample,pre_len,W,H)
        {Scaler} -- train data max/min
    """
    risk_taxi_time_data = pkl.load(open(all_data_filename, 'rb')).astype(np.float32)
    risk_taxi_time_data_small = pkl.load(open(all_data_filename_small, 'rb')).astype(np.float32)

    for i in split_and_norm_data(risk_taxi_time_data, risk_taxi_time_data_small,
                                 train_rate=train_rate,
                                 valid_rate=valid_rate,
                                 recent_prior=recent_prior,
                                 week_prior=week_prior,
                                 one_day_period=one_day_period,
                                 days_of_week=days_of_week,
                                 pre_len=pre_len):
        yield i


def split_and_norm_data_time(all_data, all_data_small,
                             contents,
                             content,
                             train_rate=0.6,
                             valid_rate=0.2,
                             recent_prior=3,
                             week_prior=4,
                             one_day_period=24,
                             days_of_week=7,
                             pre_len=1,
                             grid_len=None):
    num_of_time, channel, _, _ = all_data.shape
    num_of_time_small, channel_small, _, _ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate + valid_rate))
    all_data_small = all_data_small[0:all_data.shape[0], :, :, :]

    for index, (start, end) in enumerate(((0, train_line), (train_line, valid_line), (valid_line, num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC(all_data[start:end, :, :, :])
                scaler_small = Scaler_NYC(all_data_small[start:end, :, :, :])
            if channel == 41:
                scaler = Scaler_Chi(all_data[start:end, :, :, :])
                scaler_small = Scaler_Chi(all_data_small[start:end, :, :, :])
        norm_data = scaler.transform(all_data[start:end, :, :, :])
        norm_data_small = scaler.transform(all_data_small[start:end, :, :, :])

        grid_small_w_h = grid_len  # it should be 5(k=4) or 10(k=2).
        if channel == 48:
            feature_1 = norm_data[:, 0:1, :, :].reshape(end - start, -1)
            feature_2 = norm_data[:, 1:33, :, :].reshape(end - start, -1)
            feature_3 = norm_data[:, 33:40, :, :].reshape(end - start, -1)
            feature_4 = norm_data[:, 40:46, :, :].reshape(end - start, -1)
            feature_5 = norm_data[:, 46:48, :, :].reshape(end - start, -1)

            feature_2_4 = np.concatenate((feature_2, feature_4), axis=1).reshape(-1, 38, 20, 20)
            feature_1_3_5 = np.concatenate((feature_1, feature_3, feature_5), axis=1).reshape(-1, 10, 20, 20)

            feature_1_small = norm_data_small[:, 0:1, :, :].reshape(end - start, -1)
            feature_2_small = norm_data_small[:, 1:33, :, :].reshape(end - start, -1)
            feature_3_small = norm_data_small[:, 33:40, :, :].reshape(end - start, -1)
            feature_4_small = norm_data_small[:, 40:46, :, :].reshape(end - start, -1)
            feature_5_small = norm_data_small[:, 46:48, :, :].reshape(end - start, -1)

            feature_2_4_small = np.concatenate((feature_2_small, feature_4_small), axis=1).reshape(-1, 38,
                                                                                                   grid_small_w_h,
                                                                                                   grid_small_w_h)
            feature_1_3_5_small = np.concatenate((feature_1_small, feature_3_small, feature_5_small), axis=1).reshape(
                -1, 10, grid_small_w_h, grid_small_w_h)

        elif channel == 41:
            feature_1 = norm_data[:, 0:1, :, :].reshape(end - start, -1)
            feature_2 = norm_data[:, 1:33, :, :].reshape(end - start, -1)
            feature_4 = norm_data[:, 33:39, :, :].reshape(end - start, -1)
            feature_5 = norm_data[:, 39:41, :, :].reshape(end - start, -1)

            feature_2_4 = np.concatenate((feature_2, feature_4), axis=1).reshape(-1, 38, 20, 20)
            feature_1_3_5 = np.concatenate((feature_1, feature_5), axis=1).reshape(-1, 3, 20, 20)

            feature_1_small = norm_data_small[:, 0:1, :, :].reshape(end - start, -1)
            feature_2_small = norm_data_small[:, 1:33, :, :].reshape(end - start, -1)
            feature_4_small = norm_data_small[:, 33:39, :, :].reshape(end - start, -1)
            feature_5_small = norm_data_small[:, 39:41, :, :].reshape(end - start, -1)

            feature_2_4_small = np.concatenate((feature_2_small, feature_4_small), axis=1).reshape(-1, 38,
                                                                                                   grid_small_w_h,
                                                                                                   grid_small_w_h)
            feature_1_3_5_small = np.concatenate((feature_1_small, feature_5_small), axis=1).reshape(-1, 3,
                                                                                                     grid_small_w_h,
                                                                                                     grid_small_w_h)

        X, Y, target_time = [], [], []
        X_small, Y_small, target_time_small = [], [], []
        high_X, high_Y, high_target_time = [], [], []
        high_X_small, high_Y_small, high_target_time_small = [], [], []
        for i in range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period  # 673
            label = norm_data[t:t + pre_len, 0, :, :]
            period_list = []
            for week in range(week_prior):
                period_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature = feature_1_3_5[period_list, :, :, :]
            X.append(feature)
            Y.append(label)
            target_time.append(feature_2_4[t, :, 0, 0])

            if list(norm_data[t, 1:25, 0, 0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
                high_target_time.append(feature_2_4[t, :, 0, 0])

        for i in range(len(norm_data_small) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period  # 673
            label = norm_data_small[t:t + pre_len, 0, :, :]
            period_list = []
            for week in range(week_prior):
                period_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature_small = feature_1_3_5_small[period_list, :, :, :]

            X_small.append(feature_small)
            Y_small.append(label)
            target_time_small.append(norm_data[t, 1:33, 0, 0])

            if list(norm_data_small[t, 1:25, 0, 0]).index(1) in high_fre_hour:
                high_X_small.append(feature_small)
                high_Y_small.append(label)

                high_target_time_small.append(norm_data[t, 1:33, 0, 0])
        yield np.array(X), np.array(X_small), np.array(Y), np.array(target_time), np.array(high_X), np.array(
            high_X_small), np.array(high_Y), np.array(high_target_time), scaler


def normal_and_generate_dataset_time(
        all_data_filename,
        all_data_filename_small,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1,
        grid_len=None
):
    all_data = pkl.load(open(all_data_filename, 'rb')).astype(np.float32)
    all_data_small = pkl.load(open(all_data_filename_small, 'rb')).astype(np.float32)
    content = joblib.load(open("D:/Data/data/nyc/shikong.pkl", 'rb')).astype(np.float32)
    contents = joblib.load(open("D:/Data/data/nyc/shikong1.pkl", 'rb')).astype(np.float32)
    for i in split_and_norm_data_time(all_data, all_data_small,
                                      contents,
                                      content,
                                      train_rate=train_rate,
                                      valid_rate=valid_rate,
                                      recent_prior=recent_prior,
                                      week_prior=week_prior,
                                      one_day_period=one_day_period,
                                      days_of_week=days_of_week,
                                      pre_len=pre_len,
                                      grid_len=grid_len):
        yield i


def get_mask(mask_path):
    """
    Arguments:
        mask_path {str} -- mask filename
    
    Returns:
        {np.array} -- mask matrix，维度(W,H)
    """
    mask = pkl.load(open(mask_path, 'rb')).astype(np.float32)
    return mask


def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path
    
    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path, 'rb')).astype(np.float32)
    return adjacent


def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename
    
    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path, 'rb')).astype(np.float32)
    return grid_node_map

