#!/usr/bin/python2.7

import torch
from dynamic_model import Trainer
from help import create_log
from batch_gen_dynamic import ProperLayerBatchGenerator
import numpy as np
import os
import argparse
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', default='4s_12_pw_relu')
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='12345')
parser.add_argument('--layers', default='12')
parser.add_argument('--numstages', default='4')
parser.add_argument('--kernelsize', default='3')
parser.add_argument('--weighted', default='0')
parser.add_argument('--lambda_param', default='0.15')
parser.add_argument('--gamma', default='1')

args = parser.parse_args()

num_stages = int(args.numstages)
num_layers = int(args.layers)
kernel_size = int(args.kernelsize)
num_f_maps = 64
features_dim = 2048
if args.dataset == 'breakfast':
    bz = 4
else:
    bz = 1
lr = 0.0005
num_epochs = 50
start_training_from_epoch = 0
gamma = float(args.gamma)

# use the full temporal resolution @ 15fps
sample_rate = 1
if args.dataset == "50salads":
    sample_rate = 2

features_path = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/features/"
gt_path = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/groundTruth/"
gt_sd = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/groundTruthStartingDist/"
gt_ed = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/groundTruthEndingDist/"

mapping_file = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/mapping.txt"

model_dir = "/media/data/moellenbrok/ms-tcn/models/" + args.dataset + "/"
results_dir = "/media/data/moellenbrok/ms-tcn/results/" + args.dataset + "/"

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

weights = []
if args.weighted == '0':
    weights_tensor = [1.0 for i in range(num_classes)]
else:
    with open(args.dataset + '_median_class_weights.txt', 'r') as f:
        for line in f.read().split('\n')[:-1]:
            weight = float(line.split()[1])
            weights.append(weight)
weights_tensor = torch.as_tensor(weights, device=device)

if args.action == "train":
    model_counter = 0
    created_new_model = False
    final_model_name = args.modelname
    while not created_new_model:
        if not os.path.exists(model_dir + args.modelname):
            for i in list(args.split):
                os.makedirs(model_dir + args.modelname + '/split_' + i)
            created_new_model = True

        elif start_training_from_epoch != 0:
            created_new_model = True

        elif not os.path.exists(model_dir + args.modelname + '_' + str(model_counter)):
            for i in list(args.split):
                os.makedirs(model_dir + args.modelname + '_' + str(model_counter) + '/split_' + i)
            created_new_model = True
            final_model_name = args.modelname + '_' + str(model_counter)
        else:
            model_counter += 1

    logfile_path = model_dir + final_model_name + '/parameter_log_' + str(start_training_from_epoch) + '.txt'
    params = [final_model_name, args.dataset, args.split, args.layers, args.numstages, args.weighted, args.gamma,
              bz, lr, num_epochs]
    create_log(logfile_path, params)
    epoch_loss = np.zeros(num_epochs - start_training_from_epoch)
    epoch_acc = np.zeros(num_epochs - start_training_from_epoch)

    for i in list(args.split):
        batch_gen = ProperLayerBatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, gt_sd, gt_ed,  num_layers, 1, kernel_size)
        vid_list_file = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/splits/train.split" + i + ".bundle"
        batch_gen.read_data(vid_list_file)
        model = model_dir + final_model_name + '/split_' + i


        trainer = Trainer(num_stages, num_layers, kernel_size, num_f_maps, features_dim, num_classes, logfile_path)
        trainer.train(model, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device, gamma=gamma, start_training_epoch=start_training_from_epoch, epoch_loss_acc=[epoch_loss, epoch_acc])
    epoch_acc = epoch_acc / len(args.split)
    epoch_loss = epoch_loss / len(args.split)
    argmin_loss = np.argmin(epoch_loss) + 1 + start_training_from_epoch
    argmax_acc = np.argmax(epoch_acc) + 1 + start_training_from_epoch
    min_loss = np.amin(epoch_loss)
    max_acc = np.amax(epoch_acc)
    with open(model_dir + final_model_name + '/best_loss_acc_' + str(start_training_from_epoch) + '.txt', 'w') as g:
        g.write(str(argmin_loss) + ' ' + str(min_loss) + '\n')
        g.write(str(argmax_acc) + ' ' + str(max_acc))


if args.action == "predict":
    for i in list(args.split):
        vid_list_file_tst = "/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/splits/test.split" + i + ".bundle"
        model = model_dir + args.modelname + '/split_' + i
        results = results_dir + args.modelname + '/split_' + i
        if not os.path.exists(results):
            os.makedirs(results)
        trainer = Trainer(num_stages, num_layers,kernel_size, num_f_maps, features_dim, num_classes)
        trainer.predict(model, results, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)

if args.action == "probability":
    trainer.probs(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device,
                  sample_rate)
