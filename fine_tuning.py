# !/usr/bin/python
# coding: utf8
# @Time  : 2022/6/22 18:18
# @Author : 张晓雨
# @Email  : 1831797188@qq.com
# @Software: PyCharm

from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
import argparse

from tqdm import tqdm
import pickle
import os
from sklearn.metrics import mean_absolute_error

from torch.utils.tensorboard import SummaryWriter

from models.StarNet import StarNet
from models.RRNet import RRNet

parser = argparse.ArgumentParser()

parser.add_argument(
    '--net', choices=['StarNet', 'RRNet', 'SPEMR'], default='SPEMR',
    help='The models you need to use.',
)
parser.add_argument(
    '--mode', choices=['raw', 'pre-RNN', 'post-RNN'], default='post-RNN',
    help='The mode of the RRN embedding.',
)
parser.add_argument(
    '--list_ResBlock_inplanes', type=list, default=[4, 8, 16],
    help='The size of inplane in the RRNet residual block.'
)
parser.add_argument(
    '--n_rnn_sequence', type=int, default=5,
    help='The number of RRN sequences.'
)
parser.add_argument(
    '--path_reference_set', type=str, default='/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/',
    help='The path of the reference set.',
)
parser.add_argument(
    '--path_log', type=str, default='./model_log_SPEMR/',
    help='The path to save the model data after training.'
)
parser.add_argument(
    '--batch_size', type=int, default=256,
    help='The size of the batch.'
)
parser.add_argument(
    '--n_epochs', type=int, default=30,
    help='Number of epochs to train.'
)
parser.add_argument(
    '--add_training_noise', type=bool, default=True,
    help='Whether to add Gaussian noise with a mean of 1 and a variance of 1 during training'
)
parser.add_argument(
    '--label_list', type=list, default=['ASPCAP_Teff[K]', 'ASPCAP_Logg', 'ASPCAP_CH', 'ASPCAP_NH', 'ASPCAP_OH',
       'ASPCAP_MgH', 'ASPCAP_AlH', 'ASPCAP_SiH', 'ASPCAP_SH', 'ASPCAP_KH',
       'ASPCAP_CaH', 'ASPCAP_TiH', 'ASPCAP_CrH', 'ASPCAP_MnH', 'ASPCAP_FeH',
       'ASPCAP_NiH'],
    help='The label data that needs to be learned.'
)
parser.add_argument(
    '--noise_model', type=bool, default=True,
    help='Whether to use a model trained with noise'
)
parser.add_argument(
    '--DeepEnsemble', type=bool, default=True,
    help='Whether to use a fine-tuning model'
)


def add_noise(input_x):
    normal_noise = torch.normal(mean=torch.zeros_like(input_x), std=torch.ones_like(input_x))
    normal_prob = torch.rand_like(input_x)
    input_x[normal_prob < 0.25] += normal_noise[normal_prob < 0.25]

    return input_x


def train(args, dataset_info, train_label=['ASPCAP_Teff[K]', 'ASPCAP_Logg', 'ASPCAP_FeH'], submodel="RRNet1", model_number="SP0", cuda=True):
    if args.net == "SPEMR":
        if args.mode != "raw":
            model_name = "SPEMR(Nr=[%s]-Ns=%d)_%s" % (
                '-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.n_rnn_sequence, args.mode)
        else:
            model_name = "SPEMR(Nr=[%s])_%s" % ('-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.mode)

        net = RRNet(
            mode=args.mode,
            num_lable=len(train_label),
            list_ResBlock_inplanes=args.list_ResBlock_inplanes,
            num_rnn_sequence=args.n_rnn_sequence,
            len_spectrum=7200,
        )
        if cuda:
            net = net.to("cuda")
    elif args.net == "StarNet":
        model_name = "StarNet_%s" % args.mode

        net = StarNet(
            num_lable=len(train_label),
            mode=args.mode,
        )
        if cuda:
            net = net.to("cuda")

    if args.add_training_noise:
        model_name += "_add-noise"
        
    log_dir = args.path_log + model_name + '_' + submodel + '/' + model_number
    print("Log_dir:", log_dir)
    
    model_name += '/' + model_number
    model_path = args.path_log + model_name
    net.load_state_dict(torch.load(model_path + "/weight_best.pkl"))
    print(net)
    print("model_path:",  model_path)
    print("model_name:", model_name)

    optimizer = torch.optim.Adam(net.fc1.parameters(), lr=0.0002)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    
    writer = SummaryWriter(log_dir=log_dir)

    label_index = [args.label_list.index(i) for i in train_label]

    best_loss = np.inf
    # Iterative optimization
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        net.train()
        torch.cuda.empty_cache()

        train_mae = np.zeros(len(train_label))
        train_loss = 0.0
        # Train
        for step, (batch_x, batch_y) in enumerate(dataset_info["train_loader"]):

            if args.add_training_noise and epoch > 5:
                batch_x = add_noise(batch_x)

            batch_y = batch_y[:, label_index]
            if cuda:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
            mu, sigma = net(batch_x)
            loss = net.get_loss(batch_y, mu, sigma)

            train_loss += loss.to("cpu").data.numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(dataset_info["train_loader"]) + step + 1

            mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][
                label_index]
            batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + \
                      dataset_info["label_mean"][label_index]

            mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')
            train_mae += mae

            writer.add_scalar('Train/loss', loss.to("cpu").data.numpy(), n_iter)
            for i in range(len(train_label)):
                writer.add_scalar('Train/%s_MAE' % train_label[i], mae[i], n_iter)

        scheduler.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        train_loss /= (step + 1)
        train_mae /= (step + 1)

        torch.cuda.empty_cache()
        net.eval()

        valid_mae = np.zeros(len(label_index))
        vlaid_diff_std = np.zeros(len(label_index))
        valid_loss = 0.0
        # Valid
        for step, (batch_x, batch_y) in enumerate(dataset_info["valid_loader"]):

            with torch.no_grad():
                batch_y = batch_y[:, label_index]
                if cuda:
                    batch_x = batch_x.to("cuda")
                    batch_y = batch_y.to("cuda")
                mu, sigma = net(batch_x)
                loss = net.get_loss(batch_y, mu, sigma)

                valid_loss += loss.to("cpu").data.numpy()

                n_iter = (epoch - 1) * len(dataset_info["valid_loader"]) + step + 1

                sigma = np.sqrt(sigma.to("cpu").data.numpy()) * dataset_info["label_std"][label_index]

                mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][
                    label_index]
                batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + \
                          dataset_info["label_mean"][label_index]

                diff_std = (mu - batch_y).std(axis=0)
                sigma_mean = sigma.mean(axis=0)

                vlaid_diff_std += diff_std

                mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')

                valid_mae += mae

            writer.add_scalar('Valid/loss', loss.to("cpu").data.numpy(), n_iter)
            for i in range(len(train_label)):
                writer.add_scalar('Valid/%s_MAE' % train_label[i], mae[i], n_iter)
                writer.add_scalar('Valid/%s_diff_std' % train_label[i], diff_std[i], n_iter)
                writer.add_scalar('Valid/%s_sigma' % train_label[i], sigma_mean[i], n_iter)

        valid_loss /= (step + 1)
        valid_mae /= (step + 1)
        vlaid_diff_std /= (step + 1)

        torch.save(net.state_dict(), log_dir + '/weight_temp.pkl')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), log_dir + '/weight_best.pkl')

        print("EPOCH %d | lr %f | train_loss %.4f | valid_loss %.4f" % (epoch, lr, train_loss, valid_loss),
              "| valid_mae", valid_mae,
              "| valid_diff_std", vlaid_diff_std)


def predict(args, test_flux_path, test_label_path=None, sub_model="RRNet1", sub_data="S1"):
    def one_predict(args, test_loader, model_path):
        print(model_path)

        train_label = ['ASPCAP_Teff[K]', 'ASPCAP_Logg', 'ASPCAP_FeH'] if model_path.split("/")[-1][:2] == "SP" else ['ASPCAP_CH', 'ASPCAP_NH', 'ASPCAP_OH', 'ASPCAP_MgH', 'ASPCAP_AlH', 'ASPCAP_SiH', 'ASPCAP_SH',
'ASPCAP_KH', 'ASPCAP_CaH', 'ASPCAP_TiH',
 'ASPCAP_CrH', 'ASPCAP_MnH', 'ASPCAP_NiH']

        if args.net == "SPEMR":
            net = RRNet(
                mode=args.mode,
                num_lable=len(train_label),
                list_ResBlock_inplanes=args.list_ResBlock_inplanes,
                num_rnn_sequence=args.n_rnn_sequence,
                len_spectrum=7200,
            ).to("cuda")
        elif args.net == "StarNet":
            net = StarNet(
                num_lable=len(train_label),
                mode=args.mode,
            ).to("cuda")

        net.eval()
        net.load_state_dict(torch.load(model_path + "/weight_best.pkl"))

        output_label = np.zeros(shape=(len(test_loader.dataset), len(train_label)))
        output_label_err = np.zeros_like(output_label)
        for step, batch in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                batch_x = batch[0]

                mu, sigma = net(batch_x.to("cuda"))
                mu = mu.to("cpu").data.numpy()
                sigma = np.sqrt(sigma.to("cpu").data.numpy())

            output_label[step * test_loader.batch_size:step * test_loader.batch_size + mu.shape[0]] = mu
            output_label_err[step * test_loader.batch_size:step * test_loader.batch_size + mu.shape[0]] = sigma

        return [output_label, output_label_err]

    label_config = pickle.load(open(args.path_reference_set + sub_data + "/label_config.pkl", 'rb'))
    label_config_index = [label_config['label_list'].index(i) for i in args.label_list]

    label_mean = label_config['label_mean'][label_config_index]
    label_std = label_config['label_std'][label_config_index]
    flux_mean = label_config["flux_mean"]
    flux_std = label_config["flux_std"]

    del label_config

    X_test_torch = pickle.load(open(test_flux_path, 'rb'))
    # 流量上限处理
    X_test_torch[X_test_torch > 2.5] = 2.5
    X_test_torch[X_test_torch < -0.5] = -0.5

    X_test_torch = (X_test_torch - flux_mean) / flux_std
    X_test_torch = torch.tensor(X_test_torch, dtype=torch.float32)
    if test_label_path is not None:
        y_test_torch = pd.read_csv(test_label_path)[args.label_list].values
        y_test_torch = (y_test_torch - label_mean) / label_std
        y_test_torch = torch.tensor(y_test_torch, dtype=torch.float32)
        # print(X_test_torch.shape)
        # print(y_test_torch.shape)
        test_dataset = Data.TensorDataset(X_test_torch, y_test_torch)
    else:
        test_dataset = Data.TensorDataset(X_test_torch)

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.net == "SPEMR":
        if args.mode != "raw":
            model_name = "SPEMR(Nr=[%s]-Ns=%d)_%s" % (
                '-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.n_rnn_sequence, args.mode)
        else:
            model_name = "SPEMR(Nr=[%s])_%s" % ('-'.join([str(i) for i in args.list_ResBlock_inplanes]), args.mode)
    elif args.net == "StarNet":
        model_name = "StarNet_%s" % args.mode
    if args.noise_model:
        model_name += "_add-noise"
    model_name = model_name + '_' + sub_model
    model_path = args.path_log + model_name
    model_list = os.listdir(model_path)
    output_list_SP = []
    output_list_CA = []
    if not args.DeepEnsemble:
        if "SP0" in model_list:
            output_list_SP.append(one_predict(args, test_loader, model_path=model_path + "/SP0"))
        if "CA0" in model_list:
            output_list_CA.append(one_predict(args, test_loader, model_path=model_path + "/CA0"))
    else:
        for model in model_list:
            out = one_predict(args, test_loader, model_path=model_path + "/" + model)
            if model[:2] == "SP":
                output_list_SP.append(out)
            elif model[:2] == "CA":
                output_list_CA.append(out)

    mu_list = []
    sigma_list = []
    for i in range(min(len(output_list_SP), len(output_list_CA))):
        mu_list.append(np.hstack((output_list_SP[i][0], output_list_CA[i][0])))
        sigma_list.append(np.hstack((output_list_SP[i][1], output_list_CA[i][1])))

    del output_list_SP, output_list_CA
    mu_list = np.array(mu_list)
    sigma_list = np.array(sigma_list)

    out_mu = mu_list.mean(0)
    out_sigma = ((mu_list ** 2 + sigma_list ** 2)).mean(0) - out_mu ** 2
    out_sigma = np.sqrt(out_sigma)

    train_label = ['ASPCAP_Teff[K]', 'ASPCAP_Logg', 'ASPCAP_FeH', 'ASPCAP_CH', 'ASPCAP_NH', 'ASPCAP_OH', 'ASPCAP_MgH', 'ASPCAP_AlH', 'ASPCAP_SiH', 'ASPCAP_SH', 'ASPCAP_KH', 'ASPCAP_CaH', 'ASPCAP_TiH', 'ASPCAP_CrH',
                   'ASPCAP_MnH', 'ASPCAP_NiH']
    train_label_index = [args.label_list.index(i) for i in train_label]

    out_mu = out_mu * label_std[train_label_index] + label_mean[train_label_index]
    out_sigma *= label_std[train_label_index]

    del mu_list, sigma_list

    if test_label_path is not None:
        true_mu = pd.read_csv(test_label_path)[train_label].values

        diff_std = (true_mu - out_mu).std(axis=0)
        mae = mean_absolute_error(true_mu, out_mu, multioutput='raw_values')
        print(
            "mae:", mae,
            "diff_std", diff_std,
        )
        df = pd.read_csv(test_label_path)
        for i in range(len(train_label)):
            df["%s_%s" % (model_name, train_label[i])] = out_mu[:, i]
            df["%s_%s_err" % (model_name, train_label[i])] = out_sigma[:, i]
        df.to_csv(test_label_path[:-4] + "_%s_out.csv" % model_name, index=False)
    else:
        df = pd.DataFrame(data=None)
        for i in range(len(train_label)):
            df["%s" % train_label[i]] = out_mu[:, i]
            df["%s_err" % train_label[i]] = out_sigma[:, i]
        df.to_csv(test_flux_path[:-4] + "_%s_out.csv" % model_name, index=False)


def get_dataset_info(args, sub_data):
    label_config = pickle.load(open(args.path_reference_set + sub_data + "/label_config.pkl", 'rb'))
    label_config_index = [label_config['label_list'].index(i) for i in args.label_list]

    label_mean = label_config['label_mean'][label_config_index]
    label_std = label_config['label_std'][label_config_index]
    flux_mean = label_config["flux_mean"]
    flux_std = label_config["flux_std"]

    del label_config

    X_train_torch = (pickle.load(open(args.path_reference_set + sub_data + "/train_flux.pkl", 'rb')) - flux_mean) / flux_std
    X_valid_torch = (pickle.load(open(args.path_reference_set + sub_data + "/valid_flux.pkl", 'rb')) - flux_mean) / flux_std

    X_train_torch = torch.tensor(X_train_torch, dtype=torch.float32)
    X_valid_torch = torch.tensor(X_valid_torch, dtype=torch.float32)

    y_train_torch = pd.read_csv(args.path_reference_set + sub_data + "/train_label.csv")[args.label_list].values
    y_valid_torch = pd.read_csv(args.path_reference_set + sub_data + "/valid_label.csv")[args.label_list].values

    y_train_torch = (y_train_torch - label_mean) / label_std
    y_valid_torch = (y_valid_torch - label_mean) / label_std

    y_train_torch = torch.tensor(y_train_torch, dtype=torch.float32)
    y_valid_torch = torch.tensor(y_valid_torch, dtype=torch.float32)

    train_dataset = Data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = Data.TensorDataset(X_valid_torch, y_valid_torch)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    dataset_info = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "label_mean": label_mean,
        "label_std": label_std,
    }

    return dataset_info


if __name__ == "__main__":
    args = parser.parse_args()

#     dataset_info = get_dataset_info(args, sub_data = "S4")


#     for i in range(0, 6):
#         train(args, dataset_info=dataset_info,
#               submodel="RRNet4",
#               train_label=['ASPCAP_Teff[K]', 'ASPCAP_Logg', 'ASPCAP_FeH'],
#               model_number="SP%d" % i)
#     for i in range(0, 6):
#         train(args, dataset_info=dataset_info,
#               submodel="RRNet4",
#               train_label=['ASPCAP_CH', 'ASPCAP_NH', 'ASPCAP_OH', 'ASPCAP_MgH', 'ASPCAP_AlH', 'ASPCAP_SiH', 'ASPCAP_SH', 'ASPCAP_KH', 'ASPCAP_CaH', 'ASPCAP_TiH', 'ASPCAP_CrH', 'ASPCAP_MnH', 'ASPCAP_NiH'],
#               model_number="CA%d" % i)

#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_label.csv", sub_model="RRNet1", sub_data="refer_set")
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_label.csv", sub_model="RRNet1", sub_data="refer_set")
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_label.csv", sub_model="RRNet1", sub_data="refer_set")
    
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_label.csv", sub_model="RRNet2", sub_data="refer_set")
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_label.csv", sub_model="RRNet2", sub_data="refer_set")
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_label.csv", sub_model="RRNet2", sub_data="refer_set")
    
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_label.csv", sub_model="RRNet33", sub_data="refer_set")
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_label.csv", sub_model="RRNet33", sub_data="refer_set")
    
#     predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_flux.pkl",
#             test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_label.csv", sub_model="RRNet33", sub_data="refer_set")

    predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_flux.pkl",
            test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/train_label.csv", sub_model="RRNet4", sub_data="refer_set")
    
    predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_flux.pkl",
            test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/valid_label.csv", sub_model="RRNet4", sub_data="refer_set")
    
    predict(args, test_flux_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_flux.pkl",
            test_label_path="/home/DM2/workspace/SPEMR/data/LAMOST_DR8_APOGEE_DR17/refer_set/test_label.csv", sub_model="RRNet4", sub_data="refer_set")
    


    all_flux_path = "/home/DM2/workspace/SPEMR/data/lamost_dr8_medium_star/all_flux_data/"
#     for i in tqdm(range(1384)):
#         test_flux_path = all_flux_path + "all_not_match_valid_flux_%d.pkl" % i
#         print(test_flux_path)
#         predict(args, test_flux_path=test_flux_path, sub_model="RRNet11", sub_data="refer_set")
        
#     for i in tqdm(range(1384)):
#         test_flux_path = all_flux_path + "all_not_match_valid_flux_%d.pkl" % i
#         print(test_flux_path)
#         predict(args, test_flux_path=test_flux_path, sub_model="RRNet2", sub_data="refer_set")
        
#     for i in tqdm(range(1384)):
#         test_flux_path = all_flux_path + "all_not_match_valid_flux_%d.pkl" % i
#         print(test_flux_path)
#         predict(args, test_flux_path=test_flux_path, sub_model="RRNet33", sub_data="refer_set")


    for i in tqdm(range(1384)):
        test_flux_path = all_flux_path + "all_not_match_valid_flux_%d.pkl" % i
        print(test_flux_path)
        predict(args, test_flux_path=test_flux_path, sub_model="RRNet4", sub_data="refer_set")