#!/usr/bin/python2.7
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np

primes_plus_one_linear = [1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,
                          127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,
                          251,257,263,269,271,277,281]
primes_plus_one_exp = [1,2,3,7,13,31,61,127,251,509,1021,2039,4093,8191]


class CombinedSingleStageModel(nn.Module):
    def __init__(self,layers, num_f_maps, dim, num_classes, kernel_size):
        super(CombinedSingleStageModel, self).__init__()

        self.subnet_medium_rf = SingleStageModel(layers-2, num_f_maps, dim, num_classes, kernel_size,2)
        self.subnet_large_rf = SingleStageModel(layers, num_f_maps, dim, num_classes,kernel_size,2)
        #self.subnet_small_rf = SingleStageModel(layers-2, num_f_maps, dim, num_f_maps, kernel_size, 2)

        self.conv_out = nn.Conv1d(num_classes*2, num_classes, 1)


    def forward(self, x, mask):

        #out_s = self.subnet_small_rf(x, mask)
        out_m = self.subnet_medium_rf(x, mask)
        out_l = self.subnet_large_rf(x, mask)
        #out_m = 10*out_m / torch.norm(out_m,dim=

        out = torch.cat((out_m, out_l), dim=1)

        out = self.conv_out(out) * mask[:, 0:1, :]


        return out, out_m, out_l


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, layers, num_f_maps, dim, num_classes, kernel_size):
        super().__init__()

        #self.stage1 = SingleStageModel(int(layers), num_f_maps, dim, num_classes, kernel_size)
        self.stage1 = SingleStageModel(int(layers), num_f_maps, dim, num_classes, kernel_size)
        self.stages = nn.ModuleList([copy.deepcopy(StandardSingleStageModel(10, num_f_maps, num_classes, num_classes, kernel_size)) for s in range(num_stages-1)])


    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        #print(out.shape)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class StandardSingleStageModel(nn.Module):
    def __init__(self, layers, num_f_maps, dim, num_classes, kernel_size):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(kernel_size, 2**i, num_f_maps, num_f_maps)) for i in range(layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out



class SingleStageModel(nn.Module):
    def __init__(self, layers, num_f_maps, dim, num_classes, kernel_size, dilation_base=2, start_dilation = 0):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(kernel_size,4**i, num_f_maps, num_f_maps)) for i in range(layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out

class DilatedResidualBlockLayer(nn.Module):
    def __init__(self, kernel_size, dilation, in_channels, out_channels, block_size=1):
        super().__init__()
        #self.conv1_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)*dilation, dilation=dilation)
        self.convs_dilated = nn.ModuleList([copy.deepcopy(nn.Conv1d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)*dilation, dilation=dilation)) for i in range(block_size)])
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = x
        for conv in self.convs_dilated:
            out = conv(out) * mask[:, 0:1, :]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class DilatedResidualLayer(nn.Module):
    def __init__(self, kernel_size, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)*dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        #print(str(out.shape) + ' ' + str(x.shape))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #print(out.shape)
        return (x + out) * mask[:, 0:1, :]



class CombinedDilatedResidualLayer(nn.Module):
    def __init__(self, kernel_size, dilation, dilation2,dilation3, in_channels, out_channels):
        super(CombinedDilatedResidualLayer, self).__init__()
        self.conv_3_1 = nn.Conv1d(in_channels, out_channels, 3, padding= dilation, dilation=dilation)
        self.conv_3_2 = nn.Conv1d(in_channels, out_channels, 3, padding= dilation2, dilation=dilation2)
        self.conv_3_3 = nn.Conv1d(in_channels, out_channels, 3, padding= dilation3, dilation=dilation3)
        #self.conv_3_4 = nn.Conv1d(in_channels, out_channels, 3, padding= dilation4, dilation=dilation4)
        #self.conv_9 = nn.Conv1d(in_channels, out_channels, 9, padding= 4 * dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels*3, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = torch.cat(
            (F.relu(self.conv_3_1(x)), F.relu(self.conv_3_2(x)),F.relu(self.conv_3_3(x))),
            dim=1)
        #out = torch.cat((F.relu(self.conv_3_1(x)) ,F.relu(self.conv_3_2(x)), F.relu(self.conv_3_3(x)), F.relu(self.conv_3_4(x))), dim=1)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:,0:1,:]

class Trainer:
    def __init__(self, num_blocks, layers, kernel_size, num_f_maps, dim, num_classes, weights_tensor, logfile_path=None):
        self.model = MultiStageModel(num_blocks, layers, num_f_maps, dim, num_classes, kernel_size)  #  # #   #CombinedSingleStageModel(num_f_maps,dim,num_classes) #
        self.ce = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.log_file = logfile_path
        if not logfile_path == None:
            print(self.model, file=open(self.log_file, 'a'))

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, lambda_param=0.15, start_training_epoch=0, epoch_loss_acc=None):
        self.model.train()
        self.model.to(device)
        if start_training_epoch != 0:
            self.model.load_state_dict(torch.load(save_dir + "/epoch-" + str(start_training_epoch) + ".model"))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        epoch_losses = []
        epoch_accs = []
        for epoch in range(start_training_epoch, num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            batch_num=0
            while batch_gen.has_next():
                #print(batch_num)
                batch_num+=1
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)
                loss = 0

                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += lambda_param*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()

                total_loss = loss
                total_loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            if epoch + 1 >= 45:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            if not self.log_file == None:
                with open(self.log_file, 'a') as f:
                    f.write("\n[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                                float(correct)/total))
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

            if epoch_loss_acc != None:
                epoch_losses.append(epoch_loss / len(batch_gen.list_of_examples))
                epoch_accs.append(float(correct) / total)
        if epoch_loss_acc != None:
            epoch_loss_acc[0] += np.array(epoch_losses)
            epoch_loss_acc[1] += np.array(epoch_accs)


    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            #print(self.model)
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                #print(predicted.size())
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

    def probs(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            #print(self.model)
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                probs = predictions.squeeze()
                print(probs.size())

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name + '_probs', "w")

                for j in range(len(probs[0])):
                    #f_ptr.write(' '.join(probs[:][j]))
                    for i in range(len(probs)-1):
                        f_ptr.write(str(probs[i][j].item()) + ' ')
                    f_ptr.write(str(probs[len(probs)-1][j].item()) + '\n')

                #f_ptr.close()
