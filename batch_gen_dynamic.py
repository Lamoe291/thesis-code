#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, gt_sd, gt_ed):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt_path_starting_dist = gt_sd
        self.gt_path_ending_dist = gt_ed

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def get_start_end(self, vid):
        gt_sd = np.load(self.gt_path_starting_dist + vid.split('.')[0] + '.npy')
        gt_ed = np.load(self.gt_path_ending_dist + vid.split('.')[0] + '.npy')

        return gt_sd,gt_ed



    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_start_end = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

            start_end = self.get_start_end(vid)
            batch_start_end.append(start_end)

        length_of_sequences = list(map(len, batch_target))
        #print(length_of_sequences)
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        batch_start_end_tensor = torch.zeros(len(batch_input), 2, max(length_of_sequences), dtype=torch.float)

        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            batch_start_end_tensor[i, 0, :np.shape(batch_start_end[i])[1]] = torch.from_numpy(batch_start_end[i][0])
            batch_start_end_tensor[i, 1, :np.shape(batch_start_end[i])[1]] = torch.from_numpy(batch_start_end[i][1])

            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch_start_end_tensor

class ProperLayerBatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, gt_sd, gt_ed, num_layers, num_stages, kernel_size):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt_path_starting_dist = gt_sd
        self.gt_path_ending_dist = gt_ed
        self.num_layers = num_layers
        self.num_stages = num_stages
        self.kernel_size = kernel_size

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def get_max_distance(self, vid):
        gt_sd = np.load(self.gt_path_starting_dist + vid.split('.')[0] + '.npy')
        gt_ed = np.load(self.gt_path_ending_dist + vid.split('.')[0] + '.npy')

        return np.ndarray.max(np.array([gt_ed, gt_sd]),0)

    def get_action_lengths(self, vid):
        gt_sd = np.load(self.gt_path_starting_dist + vid.split('.')[0] + '.npy')
        gt_ed = np.load(self.gt_path_ending_dist + vid.split('.')[0] + '.npy')

        return gt_ed + gt_sd + 1

    def get_proper_layers(self, receptive_field_sizes):
        stage_cut = (self.kernel_size - 1) * (2 **self.num_layers - 1) + 1
        all_covered_frames = receptive_field_sizes <= stage_cut
        pl_raw  = np.log2((receptive_field_sizes-1)/(self.kernel_size-1)+1) * all_covered_frames.astype(float)

        #for i in range(1, self.num_stages):
        #    not_covered_frames = ~all_covered_frames
        #    lengths_still_to_cover = (receptive_field_sizes - i * (stage_cut - 1)) * not_covered_frames.astype(float)
        #    covered_frames_this_stage = (lengths_still_to_cover <= stage_cut) * not_covered_frames
        #    pl_raw_this_stage = ((np.log2(
        #        lengths_still_to_cover + 1) - 1) + i * self.num_layers) * covered_frames_this_stage.astype(float)
        #    all_covered_frames = all_covered_frames + covered_frames_this_stage
        #    pl_raw = pl_raw + pl_raw_this_stage

        pl_raw = pl_raw + (~all_covered_frames).astype(float) * (self.num_stages * self.num_layers)
        pl = np.ceil(pl_raw)
        return pl



    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_proper_layers = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

            max_dist = np.clip(self.get_max_distance(vid),a_min=1, a_max=None)
            #action_lengts = np.clip(self.get_action_lengths(vid),a_min=2, a_max=None)
            r_field_size_needed = 2 * max_dist + 1 #action_lenths
            proper_layers = self.get_proper_layers(r_field_size_needed) - 1
            batch_proper_layers.append(proper_layers[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        #print(length_of_sequences)
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        batch_pl_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)

        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            batch_pl_tensor[i, :np.shape(batch_proper_layers[i])[0]] = torch.from_numpy(batch_proper_layers[i])


            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch_pl_tensor
