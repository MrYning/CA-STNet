import torch
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import json
import argparse
import random
import sys
import os
from numpy import genfromtxt
from lib.dataloader import normal_and_generate_dataset_time, get_mask, get_grid_node_map_maxtrix
from lib.early_stop import EarlyStopping
from model.CA_STNet import GSNet
from lib.utils import mask_loss, compute_loss, predict_and_evaluate

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

city = 'nyc'
grid_small = 10  # it should be 5(k=4) or 10(k=2).
single_model = False
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=f"./config/{city}/GSNet_{city}_Config.json",
                    help='configuration file')
parser.add_argument("--gpus", type=str, default='0', help="test program")
parser.add_argument("--test", action="store_true", help="test program")

args = parser.parse_args()
print(args.config)
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device=", device)

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']
all_data_filename_small = config['all_data_filename_small']
trans_file = config['trans']

mask_filename = config['mask_filename']
grid_node_filename = config['grid_node_filename']
grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)
num_of_vertices = grid_node_map.shape[1]

patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']
data_type = config['data_type']


def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             high_test_loader,
             risk_mask,
             trainer,
             early_stop,
             device,
             scaler,
             data_type=data_type
             ):
    global_step = 1
    for epoch in range(1, training_epoch + 1):
        trans = genfromtxt(trans_file, delimiter=',').astype(np.float32)
        trans = torch.tensor(trans)
        net.train()
        for train_feature, train_feature_small, target_time, train_label in train_loader:
            trans, train_feature, train_feature_small, target_time, train_label = trans.to(device), train_feature.to(
                device), train_feature_small.to(device), target_time.to(device), train_label.to(device)
            l = mask_loss(net(train_feature, train_feature_small, target_time, trans), train_label, risk_mask,
                          data_type=data_type)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            global_step += 1

        val_loss = compute_loss(net, val_loader, risk_mask, trans_file, grid_node_map, global_step - 1, device,
                                data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' % (global_step - 1, epoch, val_loss), flush=True)

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                predict_and_evaluate(net, test_loader, risk_mask, trans_file, grid_node_map, global_step - 1, scaler,
                                     device)
            high_test_rmse, high_test_recall, high_test_map, _, _ = \
                predict_and_evaluate(net, high_test_loader, risk_mask, trans_file, grid_node_map, global_step - 1,
                                     scaler, device)
            print(
                'global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,hihg test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                % (global_step - 1, epoch, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                   high_test_map), flush=True)

        early_stop(val_loss, test_rmse, test_recall, test_map,
                   high_test_rmse, high_test_recall, high_test_map,
                   test_inverse_trans_pre, test_inverse_trans_label)
        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s" % (global_step, epoch), flush=True)

            print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                  % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map), flush=True)
            print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                  % (early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map), flush=True)

            break

    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map


def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_of_gru_layers = config['num_of_gru_layers']
    gru_hidden_size = config['gru_hidden_size']
    gcn_num_filter = config['gcn_num_filter']

    loaders = []
    scaler = ""
    train_data_shape = ""
    graph_feature_shape = ""
    for idx, (x, x_small, y, target_times, high_x, high_x_small, high_y, high_target_times, scaler) in enumerate(
            normal_and_generate_dataset_time(
                all_data_filename,
                all_data_filename_small,
                train_rate=train_rate,
                valid_rate=valid_rate,
                recent_prior=recent_prior,
                week_prior=week_prior,
                one_day_period=one_day_period,
                days_of_week=days_of_week,
                pre_len=pre_len,
                grid_len=grid_small)):

        if args.test:
            x = x[:100]
            x_small = x_small[:100]
            y = y[:100]
            target_times = target_times[:100]
            high_x = high_x[:100]
            high_x_small = high_x_small[:100]
            high_y = high_y[:100]
            high_target_times = high_target_times[:100]
        if 'nyc' in all_data_filename:
            graph_x = x[:, :, [0, 1, 2], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 1, 2], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)
        if 'chicago' in all_data_filename:
            graph_x = x[:, :, [0, 1, 2], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 1, 2], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)
        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape),
              "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape
        loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(x_small),
                torch.from_numpy(target_times),
                torch.from_numpy(y)
            ),
            batch_size=batch_size,
            shuffle=(idx == 0)
        ))
        if idx == 2:
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_x_small),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            )
    train_loader, val_loader, test_loader = loaders

    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    GSNet_Model = GSNet(
        train_data_shape[2],
        num_of_gru_layers,
        seq_len,
        pre_len,
        gru_hidden_size,
        time_shape[1],
        north_south_map,
        west_east_map,
        city,
        single_model
    )
    GSNet_Model.to(device)

    num_of_parameters = 0
    for name, parameters in GSNet_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(GSNet_Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience, delta=delta)

    risk_mask = get_mask(mask_filename)

    best_mae, best_mse, best_rmse = training(
        GSNet_Model,
        training_epoch,
        train_loader,
        val_loader,
        test_loader,
        high_test_loader,
        risk_mask,
        trainer,
        early_stop,
        device,
        scaler,
        data_type=data_type
    )
    return best_mae, best_mse, best_rmse


if __name__ == "__main__":
    main(config)
