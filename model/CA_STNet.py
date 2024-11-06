import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=10):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class D2_S_atten(nn.Module):

    def __init__(self, in_dim, activation=None):
        super(D2_S_atten, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 3), padding=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 3), padding=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 3), padding=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, width, height = x.size()
        q = torch.unsqueeze(self.query_conv(x).view(batchsize, C, width * height), 3)
        k = torch.unsqueeze(self.key_conv(x).view(batchsize, C, width * height), 2)
        v = torch.unsqueeze(self.value_conv(x).view(batchsize, C, width * height), 3)

        energy = torch.matmul(q, k)

        out = torch.matmul(energy, v)

        out = torch.squeeze(out).view(batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class D3_S_atten(nn.Module):

    def __init__(self, in_dim, activation=None):
        super(D3_S_atten, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.se = SELayer(channel=10)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, T, width, height = x.size()
        q = torch.unsqueeze(self.query_conv(x).view(batchsize, C, T, width * height), 4)
        k = torch.unsqueeze(self.key_conv(x).view(batchsize, C, T, width * height), 3)
        v = torch.unsqueeze(self.value_conv(x).view(batchsize, C, T, width * height), 4)

        energy = torch.matmul(q, k)
        out = torch.matmul(energy, v)

        out = torch.squeeze(out).view(batchsize, C, T, width, height)

        out = self.gamma * out + x
        return out


class D3_T_atten(nn.Module):

    def __init__(self, in_dim, activation=None):
        super(D3_T_atten, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.se = SELayer(channel=10)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, T, width, height = x.size()
        q = self.query_conv(x).view(batchsize, C, T, width * height)
        k = self.key_conv(x).permute(0, 1, 3, 4, 2).contiguous().view(batchsize, C, width * height, T)  # (B,C,N,T)
        v = self.value_conv(x).view(batchsize, C, T, width * height)

        energy = torch.matmul(q, k)
        out = torch.matmul(energy, v)
        out = out.view(batchsize, C, T, width, height)
        out = self.gamma * out + x
        return out


class STGeoModule(nn.Module):
    def __init__(self, grid_in_channel, num_of_gru_layers, seq_len,
                 gru_hidden_size, num_of_target_time_feature, single_model):
        super(STGeoModule, self).__init__()
        self.TD = nn.Conv3d(in_channels=grid_in_channel, out_channels=grid_in_channel, kernel_size=(1, 3, 3),
                            padding=(0, 1, 1))
        self.TD2 = nn.Conv3d(in_channels=2 * grid_in_channel, out_channels=grid_in_channel, kernel_size=(1, 1, 1))
        self.TD3 = nn.Conv3d(in_channels=grid_in_channel, out_channels=grid_in_channel, kernel_size=(7, 3, 3),
                             padding=(0, 1, 1))
        self.WD2 = nn.Conv2d(in_channels=2 * grid_in_channel, out_channels=grid_in_channel, kernel_size=1)

        self.ST_B = nn.Sequential(
            D3_S_atten(grid_in_channel),
            nn.Conv3d(in_channels=grid_in_channel, out_channels=grid_in_channel, kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            # nn.BatchNorm3d(grid_in_channel),
            nn.ReLU(),
            D3_T_atten(grid_in_channel),
        )

        self.D2S_B = nn.Sequential(
            D2_S_atten(grid_in_channel),
            nn.Conv2d(in_channels=grid_in_channel, out_channels=grid_in_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            D2_S_atten(grid_in_channel),
            nn.Conv2d(in_channels=grid_in_channel, out_channels=grid_in_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

        self.D1S_B = nn.Sequential(
            D2_S_atten(grid_in_channel),
            nn.Conv2d(in_channels=grid_in_channel, out_channels=grid_in_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )

        self.grid_att_fc = nn.Linear(in_features=grid_in_channel, out_features=gru_hidden_size)
        self.grid_att_fc1 = nn.Linear(in_features=gru_hidden_size, out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=1)
        self.grid_att_softmax = nn.Softmax(dim=-1)

    def forward(self, grid_input, grid_input_small, target_time_feature, trans):
        batch_size, T, D, W, H = grid_input.shape
        b, d, t, w, h = grid_input_small.shape
        grid_input = grid_input.permute(0, 2, 1, 3, 4)
        grid_input_small = grid_input_small
        grid_input = self.TD(grid_input)  # (B,D,T,W,H)
        grid_input_small = self.TD(grid_input_small)  # (B,D,T,W,H)

        grid_input = self.ST_B(grid_input)
        grid_input_small = self.ST_B(grid_input_small)

        conv_kk = self.TD3(grid_input)
        conv_kk_small = self.TD3(grid_input_small)

        grid_input_small1 = conv_kk_small.permute(0, 2, 1, 3, 4).contiguous().view(b, d, -1)
        grid_input_small_trans = F.relu(torch.matmul(grid_input_small1, trans)) \
            .view(batch_size, -1, D, W, H).permute(0, 2, 1, 3, 4)

        conv = torch.stack((conv_kk, grid_input_small_trans), dim=2).view(batch_size, 2 * D, -1, W, H)
        conv = self.TD2(conv)

        D3conv_output = conv.permute(0, 2, 1, 3, 4).contiguous().view(-1, D, W, H)
        D3conv_output_small = conv_kk_small.permute(0, 2, 1, 3, 4).contiguous().view(-1, d, w, h)

        D2conv_output = D3conv_output
        D2conv_output_small = D3conv_output_small

        D2conv_output = self.D2S_B(D2conv_output)
        D2conv_output_small = self.D2S_B(D2conv_output_small)
        D2conv_output = self.D1S_B(D2conv_output)
        D2conv_output_small = self.D1S_B(D2conv_output_small)

        D2conv_output_small = D2conv_output_small.contiguous().view(b, d, -1)
        grid_input_small_trans1 = F.relu(torch.matmul(D2conv_output_small, trans)).view(batch_size, D, W, H)

        conv1 = torch.stack((D2conv_output, grid_input_small_trans1), dim=2).view(batch_size, 2 * D, W, H)
        conv_kk1 = self.WD2(conv1)

        conv_output = conv_kk1.view(batch_size, -1, D, W, H) \
            .permute(0, 3, 4, 1, 2) \
            .contiguous() \
            .view(-1, 1, D)
        grid_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, W * H, 1).view(batch_size * W * H, -1)
        gru_output = torch.squeeze(self.grid_att_fc(conv_output))
        grid_att_fc1_output = self.grid_att_fc1(gru_output)
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)

        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output + grid_att_fc2_output))
        grid_att_score = grid_att_score.view(batch_size * W * H, -1, 1)
        gru_output1 = torch.unsqueeze(gru_output, 1)
        grid_output = torch.sum(gru_output1 * grid_att_score, dim=1)

        grid_output = grid_output.view(batch_size, W, H, -1).permute(0, 3, 1, 2).contiguous()

        return grid_output


class GSNet(nn.Module):
    def __init__(
            self,
            grid_in_channel,
            num_of_gru_layers,
            seq_len,
            pre_len,
            gru_hidden_size,
            num_of_target_time_feature,
            north_south_map,
            west_east_map,
            city,
            single_model
    ):

        super(GSNet, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.single_model = single_model
        print(f'single_model: {self.single_model}')
        if city == 'nyc':
            self.start_conv = nn.Conv3d(in_channels=10,
                                        out_channels=10,
                                        kernel_size=(1, 1, 1))
            self.start_conv_cluster = nn.Conv3d(in_channels=10,
                                                out_channels=10,
                                                kernel_size=(1, 1, 1))
        else:
            self.start_conv = nn.Conv3d(in_channels=3,
                                        out_channels=3,
                                        kernel_size=(1, 1, 1))
            self.start_conv_cluster = nn.Conv3d(in_channels=3,
                                                out_channels=3,
                                                kernel_size=(1, 1, 1))
        self.st_geo_module = STGeoModule(grid_in_channel, num_of_gru_layers, seq_len, gru_hidden_size,
                                         num_of_target_time_feature, self.single_model)

        fusion_channel = 16
        self.grid_weigth = nn.Conv2d(in_channels=gru_hidden_size, out_channels=fusion_channel, kernel_size=1)
        self.output_layer = nn.Linear(fusion_channel * north_south_map * west_east_map,
                                      pre_len * north_south_map * west_east_map)
        self.W = nn.parameter.Parameter(
            torch.randn(8, 7, 10, 400),
            requires_grad=True,
        )
        self.hjhj = nn.Conv3d(in_channels=2 * grid_in_channel, out_channels=grid_in_channel, kernel_size=1)

    def forward(self, grid_input, grid_input_small, target_time_feature, trans):

        batch_size1, T, D, f_N, f_L = grid_input.shape
        batch_size, T, D, c_N, c_L = grid_input_small.shape
        grid_input = grid_input.permute(0, 2, 1, 3, 4)
        grid_input_small = grid_input_small.permute(0, 2, 1, 3, 4)
        f_output = self.start_conv(grid_input)
        c_in = self.start_conv_cluster(grid_input_small)
        c_output = c_in.permute(0, 2, 1, 3, 4)
        c_graph_output = c_output.contiguous().view(batch_size * T, D, -1)
        trans = trans[:c_graph_output.shape[2], :]
        cf_out = F.relu(torch.matmul(c_graph_output, trans)).view(batch_size, T, D, f_N, f_L).permute(0, 2, 1, 3, 4)

        f1_output = torch.stack((f_output, cf_out), dim=2).view(batch_size1, 2 * D, T, f_N, f_L)
        f2_output = self.hjhj(f1_output).permute(0, 2, 1, 3, 4)

        grid_output = self.st_geo_module(f2_output, grid_input_small, target_time_feature, trans)
        grid_output = self.grid_weigth(grid_output)
        fusion_output = grid_output.view(batch_size, -1)
        final_output = self.output_layer(fusion_output).view(batch_size, -1, self.north_south_map, self.west_east_map)

        return final_output
