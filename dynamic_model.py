import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import time

primes_plus_one_linear = [1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,
                          127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,
                          251,257,263,269,271,277,281]
primes_plus_one_exp = [1,2,3,7,13,31,61,127,251,509,1021,2039,4093,8191]

class DilatedResidualLayer(nn.Module):
    def __init__(self,kernel_size, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=int((kernel_size-1)/2)*dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class OpenSingleStageModel(nn.Module):
    def __init__(self, layers,kernel_size, num_f_maps, dim, num_classes, dilation_base=2):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(kernel_size, dilation_base**i, num_f_maps, num_f_maps)) for i in range(layers)])
        #self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = self.layers[0](out, mask)
        outputs = out.unsqueeze(0)
        for i in range(1,len(self.layers)):
            out = self.layers[i](out, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        #helper_out = (self.conv_out(out) * mask[:, 0:1, :]).unsqueeze(0)
        return outputs

class SingleStageModel(nn.Module):
    def __init__(self, layers,kernel_size, num_f_maps, dim, num_classes, dilation_base=2):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(kernel_size,dilation_base**i, num_f_maps, num_f_maps)) for i in range(layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = self.layers[0](out, mask)
        outputs = out.unsqueeze(0)
        for i in range(1,len(self.layers)):
            out = self.layers[i](out, mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        helper_out = (self.conv_out(out) * mask[:, 0:1, :]).unsqueeze(0)
        return outputs, helper_out

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()

        self.stage1 = SingleStageModel(layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(layers, num_f_maps, num_f_maps, num_classes)) for s in range(num_stages-1)])


    def forward(self, x, mask):
        stage_outputs, helper_out = self.stage1(x, mask)
        all_helper_outputs = helper_out
        all_outputs = stage_outputs
        out = stage_outputs[-1] #helper_out[-1]
        for s in self.stages:
            stage_outputs, stage_helper_out = s(out, mask)
            all_outputs = torch.cat((all_outputs, stage_outputs), dim=0)
            all_helper_outputs = torch.cat((all_helper_outputs, stage_helper_out), dim=0)
            out = stage_outputs[-1] #stage_helper_out[-1]

        return all_outputs, all_helper_outputs

class PLPredictionLayer(nn.Module):
    def __init__(self, num_f_maps, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.fuse_features = nn.Conv1d(num_layers*num_f_maps, num_f_maps, 1)
        self.fc3 = nn.Linear(num_f_maps,num_f_maps)
        self.fc4 = nn.Linear(num_f_maps,num_f_maps)
        self.fc5 = nn.Linear(num_f_maps,num_layers)

        #self.pl_calc = ProperLayerCalc(num_stages, num_layers, num_f_maps)



    def forward(self, x, mask):

        mean_activation = (1.0 / (self.num_layers)) * torch.sum(x, dim=0)

        'transpose s.t. num_f_maps is last dim'
        out = F.relu(self.fc3(F.relu(mean_activation.transpose(1,2))))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        pl_out = out.transpose(1,2) * mask[:, 0:1, :]


        return pl_out

class ActionClassificationLayer(nn.Module):
    def __init__(self, num_f_maps, num_classes):
        super().__init__()
        self.num_f_maps = num_f_maps
        #self.conv_1x1 = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fc1 = nn.Linear(num_f_maps,num_f_maps)
        self.fc2 = nn.Linear(num_f_maps, num_f_maps)
        self.conv_fuse = nn.Conv1d(num_f_maps*2,num_classes,1)
        self.conv_helper = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask, pl):
        pl_indexing = pl.expand(-1,self.num_f_maps,-1).unsqueeze(0)

        selected_layer_features = torch.gather(x,0,pl_indexing).squeeze(0)
        class_out = F.relu(self.fc1(x[-1].transpose(1, 2)))
        class_out = self.fc2(class_out).transpose(1,2)
        fused = self.conv_fuse(torch.cat((class_out, selected_layer_features), dim=1))


        return fused * mask[:, 0:1, :]#, helper_out

class ActionClassificationWeightedByProbLayer(nn.Module):
    def __init__(self, num_f_maps, num_classes):
        super().__init__()
        self.num_f_maps = num_f_maps
        # self.conv_1x1 = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fc1 = nn.Linear(num_f_maps, num_f_maps)
        self.fc2 = nn.Linear(num_f_maps, num_f_maps)
        self.conv_fuse = nn.Conv1d(num_f_maps * 2, num_classes, 1)
        self.conv_helper = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x, mask, pl):
        probs = F.softmax(pl, dim=1).unsqueeze(0).transpose(0,2)
        weighted_activations = torch.sum(x*probs, dim=0)

        class_out = F.relu(self.fc1(x[-1].transpose(1, 2)))
        class_out = self.fc2(class_out).transpose(1, 2)
        fused = self.conv_fuse(torch.cat((class_out, weighted_activations), dim=1))


        return fused * mask[:, 0:1, :]





class ProperLayerSelectionModel(nn.Module):
    def __init__(self, layers,kernel_size, num_f_maps, num_classes, dim):
        super().__init__()
        self.dilated_model = SingleStageModel(layers,kernel_size, num_f_maps, dim, num_classes)
        self.pl_layer = PLPredictionLayer(num_f_maps, layers)
        self.c_layer = ActionClassificationWeightedByProbLayer(num_f_maps, num_classes)
        #self.c_layer = ActionClassificationLayer(num_f_maps, num_classes)

    def forward(self, x, mask, pl_gt):
            out, helper_out = self.dilated_model(x,mask)
            #out = self.dilated_model(x, mask)
            #pl = self.pl_layer(x,mask)
            pl = self.pl_layer(out, mask)

            #if self.training:
             #   class_pred_out = self.c_layer(out,mask,pl_gt)
            #else:
            #reversed_onehot_pl = torch.argmax(pl, dim=1).unsqueeze(1)
            class_pred_out = self.c_layer(out, mask, pl)


            return class_pred_out, pl, helper_out




class StandardSingleStageModel(nn.Module):
    def __init__(self, layers, kernel_size, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(kernel_size, 2 ** i, num_f_maps, num_f_maps)) for i in range(layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class PG_PR_Model(nn.Module):
    def __init__(self, num_stages, layers,kernel_size, num_f_maps, dim, num_classes):
        super().__init__()

        self.prediction_generation = ProperLayerSelectionModel(layers,kernel_size,num_f_maps,num_classes,dim)
        self.prediction_refinement_stages = nn.ModuleList([copy.deepcopy(StandardSingleStageModel(10,3, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])


    def forward(self, x, mask, pl_gt):
        out, pl, helper = self.prediction_generation(x, mask, pl_gt) #
        outputs = out.unsqueeze(0)
        #print(out.size())
        for s in self.prediction_refinement_stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            #print(out.size())
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs, pl.unsqueeze(0), helper.unsqueeze(0)

class Trainer:
    def __init__(self, num_stages, layers, kernel_size, num_f_maps, dim, num_classes, log_file_path=None):

        self.model =PG_PR_Model(num_stages, layers, kernel_size, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.num_layers = layers
        self.num_stages = num_stages
        self.kernel_size = kernel_size
        self.log_file = log_file_path
        if not log_file_path == None:
            print(self.model, file=open(self.log_file, 'a'))

    def get_proper_layers(self, receptive_field_sizes):
        stage_cut = (self.kernel_size - 1) * (2 **self.num_layers - 1) + 1# 2 ** (self.num_layers + 1) - 1
        all_covered_frames = receptive_field_sizes <= stage_cut
        pl_raw = np.log2((receptive_field_sizes-1)/(self.kernel_size-1)+1) * all_covered_frames.astype(float) #(np.log2(receptive_field_sizes + 1) - 1) * all_covered_frames.astype(float)


        pl_raw = pl_raw + (~all_covered_frames).astype(float) *  self.num_layers
        pl = np.ceil(pl_raw)
        return pl

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, gamma, start_training_epoch=0, epoch_loss_acc=None):
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
            batch_nr = 1
            correct_pl = 0
            t_epoch_start = time.clock()
            while batch_gen.has_next():
                t1 = time.clock()
                #print("Batch %d" % batch_nr)
                batch_nr += 1
                batch_input, batch_target, mask, pl_target = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask, pl_target = batch_input.to(device), batch_target.to(device), mask.to(device), pl_target.to(device)
                pl_target_for_indexing = (pl_target.unsqueeze(1).float() * mask[:, 0:1, :]).long()
                optimizer.zero_grad()
                label_predictions, pl_predictions, helper_out = self.model(batch_input, mask, pl_target_for_indexing)

                #print(pl_predictions.size())
                loss = 0
                helper_loss = 0
                pl_loss = 0
                for p in label_predictions:
                    #print(p.size())
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                    self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                    max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()


                total_loss = loss + pl_loss + helper_loss
                total_loss.backward()
                optimizer.step()

                _, predicted_label = torch.max(label_predictions[-1].data, 1)
                correct += ((predicted_label == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                #print(pl_predictions.data)
                _, predicted_pl = torch.max(pl_predictions[-1].data, 1)

                correct_pl += ((predicted_pl == pl_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                t2 = time.clock()
                #print('time: %f \n' % (t2-t1))

            batch_gen.reset()
            if epoch+1>=45:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            if not self.log_file == None:
                with open(self.log_file, 'a') as f:
                    f.write("\n[epoch %d]: epoch loss = %f,   acc = %f,   pl_acc = %f" % (
                    epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                    float(correct) / total, float(correct_pl)/total))
            t_epoch_end = time.clock()
            print("[epoch %d]: epoch loss = %f,   acc = %f,   pl_acc = %f,   time = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), \
                                                                                   float(correct) / total, float(correct_pl)/total, t_epoch_end-t_epoch_start))
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
            # print(self.model)
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                dataset = features_path.split('/')[-3]
                gt_sd = "/media/data/moellenbrok/ms-tcn/data/" + dataset + "/groundTruthStartingDist/"
                gt_ed = "/media/data/moellenbrok/ms-tcn/data/" + dataset + "/groundTruthEndingDist/"
                starts = np.load(gt_sd + vid.split('.')[0] + '.npy')
                ends = np.load(gt_ed + vid.split('.')[0] + '.npy')
                receptive_field_sizes = 2* np.clip(np.ndarray.max(np.array([starts, ends]), 0),a_min=1,a_max=None) +1
                #pl = (torch.Tensor(self.get_proper_layers(receptive_field_sizes)).unsqueeze(0).unsqueeze(0).to(device=device) - 1).long()
                pl = np.clip(self.get_proper_layers(receptive_field_sizes), a_min=None,a_max=self.num_layers) - 1
                pl = pl[::sample_rate]
                pl_tensor = torch.tensor(pl, dtype=torch.long)
                pl_tensor.unsqueeze_(0)
                pl_tensor.unsqueeze_(0)
                pl_tensor=pl_tensor.to(device)
                #pl_target_for_indexing = (pl_target.unsqueeze(1).float() * mask[:, 0:1, :]).long()
                #print(pl_tensor.size())

                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                #print(input_x.size())
                label_predictions, pl_predictions, _ = self.model(input_x, torch.ones(input_x.size(), device=device), pl_tensor)
                #print(label_predictions.size())
                _, label_predicted = torch.max(label_predictions[-1].data, 1)
                label_predicted = label_predicted.squeeze()
                recognition = []
                for i in range(len(label_predicted)):
                    #print(i)
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        label_predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

                _, pl_predicted = torch.max(pl_predictions[-1].data, 1)
                pl_predicted = pl_predicted + 1
                pl_predicted = pl_predicted.squeeze()
                recognition = []
                for i in range(len(pl_predicted)):
                    #print([pl_predicted[i].item()] * sample_rate)
                    recognition = np.concatenate((recognition, [str(pl_predicted[i].item())] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name + '_pl', "w")
                f_ptr.write("### Frame level proper layer recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()