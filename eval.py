#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
from comp_avg_length import get_labels_from_map


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def f1_score(tp,fp,fn):
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    f1 = 2.0 * (precision * recall) / (precision + recall)

    f1 = np.nan_to_num(f1) * 100
    return f1

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def comp_acc(correct, total):
    if total != 0:
        return 100 * float(correct) / total
    else:
        return 'NA'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="breakfast")
    parser.add_argument('--split', default='1234')
    parser.add_argument('--modelname', default='model')


    args = parser.parse_args()




    splits = list(args.split)

    ground_truth_path = "/media/data/moellenbrok/ms-tcn/data/"+args.dataset+"/groundTruth/"


    labels = get_labels_from_map("/media/data/moellenbrok/ms-tcn/data/" + args.dataset + "/mapping.txt")


    for split in splits:
        recog_path = "/media/data/moellenbrok/ms-tcn/results/"+args.dataset+"/"+args.modelname+"/split_"+ split +"/"
        file_list = "/media/data/moellenbrok/ms-tcn/data/"+args.dataset+"/splits/test.split"+ split +".bundle"

        list_of_videos = read_file(file_list).split('\n')[:-1]

        split_correct = 0
        split_total = 0
        split_edit = 0

        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        # will contain relation of correctly classified frames vs total frames per label for current split
        split_label_correct_total = {}
        for label in labels:
            split_label_correct_total[label] = [0,0]


        for vid in list_of_videos:
            gt_file = ground_truth_path + vid
            gt_content = read_file(gt_file).split('\n')[0:-1]
        
            recog_file = recog_path + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

            for i in range(len(gt_content)):
                split_label_correct_total[gt_content[i]][1]+= 1
                split_total += 1
                if gt_content[i] == recog_content[i]:
                    split_label_correct_total[gt_content[i]][0] += 1
                    split_correct += 1

            split_edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1



        with open( "/media/data/moellenbrok/ms-tcn/results/"+args.dataset+"/"+args.modelname+"/evaluation_split" + split + ".txt","w+") as f:
                split_acc = comp_acc(split_correct, split_total)
                for i in range(len(overlap)):
                    score = f1_score(tp[i],fp[i],fn[i])
                    f.write('F1@' + str(overlap[i]) + ' ' + str(score) + '\n')
                f.write('\n')
                edit = ((1.0*split_edit)/len(list_of_videos))
                f.write('Edit ' + str(edit) + '\n')
                f.write('\n')
                f.write('Acc_over_all ' + str(split_acc) + '\n')

                for label in labels:
                    acc = comp_acc(split_label_correct_total[label][0], split_label_correct_total[label][1])
                    f.write(label + ' ' + str(acc) + '\n')

    count_available_labels = {}
    sum_eval = {}

    count_available_labels['Acc_over_all'] = 0
    sum_eval['Acc_over_all'] = 0

    count_available_labels['F1@0.1'] = 0
    count_available_labels['F1@0.25'] = 0
    count_available_labels['F1@0.5'] = 0
    count_available_labels['Edit'] = 0

    sum_eval['F1@0.1'] = 0
    sum_eval['F1@0.25'] = 0
    sum_eval['F1@0.5'] = 0
    sum_eval['Edit'] = 0

    for label in labels:
        count_available_labels[label] = 0
        sum_eval[label] = 0

    for i in splits:
        with open(
                "/media/data/moellenbrok/ms-tcn/results/" + args.dataset + "/" + args.modelname + "/evaluation_split" + i + ".txt",
                'r') as f:
            for line in f.read().split('\n')[:-1]:
                if line != '':
                    data = line.split(' ')
                    label = data[0]
                    val = data[1]
                    if val != 'NA':
                        count_available_labels[label] += 1
                        sum_eval[label] += float(val)

    avg_eval = [0 for i in range(len(labels) + 5)]
    avg_eval[0] = sum_eval['F1@0.1'] / count_available_labels['F1@0.1']
    avg_eval[1] = sum_eval['F1@0.25'] / count_available_labels['F1@0.25']
    avg_eval[2] = sum_eval['F1@0.5'] / count_available_labels['F1@0.5']
    avg_eval[3] = sum_eval['Edit'] / count_available_labels['Edit']
    avg_eval[4] = sum_eval['Acc_over_all'] / count_available_labels['Acc_over_all']
    for i in range(len(labels)):
        avg_eval[i + 5] = sum_eval[labels[i]] / count_available_labels[labels[i]]

    with open(
            "/media/data/moellenbrok/ms-tcn/results/" + args.dataset + "/" + args.modelname + "/average_evaluation" + ".txt",
            'w+') as f:
        f.write('F1@0.1 ' + str(avg_eval[0]) + '\n')
        f.write('F1@0.25 ' + str(avg_eval[1]) + '\n')
        f.write('F1@0.5 ' + str(avg_eval[2]) + '\n\n')
        f.write('Edit ' + str(avg_eval[3]) + '\n\n')
        f.write('Acc_over_all ' + str(avg_eval[4]) + '\n')
        for i in range(len(labels)):
            f.write(labels[i] + ' ' + str(avg_eval[i + 5]) + '\n')
    print(avg_eval)

if __name__ == '__main__':
    main()
