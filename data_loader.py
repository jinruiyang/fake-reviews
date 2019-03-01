import nltk
import pickle
import numpy as np
import codecs
import random
import os
from os import listdir

def getint(name):
    num, base = name.split('.')
    return int(num)

def load_data(file_lists, fractions = [0.7, 0.1], random_seed = 123456, label_mode = 1, over_sampling = False):
    if label_mode > 2:
        raise RuntimeError('Do not have label mode {mode}'.format(mode = label_mode))
    if len(fractions) != 2 or np.sum(fractions) >= 1:
        raise RuntimeError('fractions error')
    fractions[1] = fractions[0] + fractions[1]
    data_set = []
    labels = []
    for f in file_lists:
        file_in = open(f, 'rb')
        data = pickle.load(file_in)
        file_in.close()
        category_name = f.split('.')[0]
        if category_name == 'bias':
            raw_txt = './bert_feature/bias_raw.txt'
            bert_feature = './bert_feature/numpy_features/bias/'
        elif category_name == 'non_bias':
            raw_txt = './bert_feature/non_bias_raw.txt'
            bert_feature = './bert_feature/numpy_features/non_bias/'
        elif category_name == 'moderate_bias':
            raw_txt = './bert_feature/moderate_bias_raw.txt'
            bert_feature = './bert_feature/numpy_features/moderate_bias/'
        bert_files = listdir(bert_feature)
        bert_files.sort(key=getint)
        with open(raw_txt) as raw_sentences:
            for review, y in zip(raw_sentences, bert_files):
                bert = np.load(bert_feature + y)
                review=review.strip()
                label = data[review]
                # print(label)
                # print(y)
                review_length = len(review.split(' '))
                # data_set.append(bert/review_length)
                data_set.append(bert)
                labels.append(label)

    random.seed(random_seed)
    lists = list(zip(data_set, labels))
    random.shuffle(lists)
    data_set, labels = zip(*lists)

    labels = list(map(int, labels))
    if label_mode == 2:
        labels = [x if x!=2 else 1 for x in labels]

    training_set = data_set[0:int(len(data_set)*fractions[0])]
    training_labels = labels[0:int(len(data_set)*fractions[0])]

    if over_sampling:
        non_bias_data = []
        non_bias_label = []
        for data, l in zip(training_set, training_labels):
            if l == 0:
                non_bias_data.append(data)
                non_bias_label.append(l)
        training_set += tuple(non_bias_data)
        training_labels += tuple(non_bias_label)


    validation_set = data_set[int(len(data_set)*fractions[0]):int(len(data_set)*fractions[1])]
    validation_labels = labels[int(len(data_set)*fractions[0]):int(len(data_set)*fractions[1])]
    testing_set = data_set[int(len(data_set)*fractions[1]):]
    testing_labels = labels[int(len(data_set)*fractions[1]):]

    assert len(training_set) == len(training_labels) and len(validation_set) == len(validation_labels) and len(testing_set) == len(testing_labels)

    print('---------------------summary of dataset-------------------')
    print('{num_train} reviews of training data'.format(num_train = len(training_set)))
    print('{num_val} reviews of validation data'.format(num_val = len(validation_set)))
    print('{num_test} reviews of testing data'.format(num_test = len(testing_set)))
    print('Over Sampling: {over_sampling}'.format(over_sampling = over_sampling))
    print('---------------------end of summary-------------------')

    return training_set, validation_set, testing_set, training_labels, validation_labels, testing_labels


if __name__ == '__main__':
    a = load_data(['non_bias.pkl', 'bias.pkl', 'moderate_bias.pkl'], [0.7,0.1],label_mode=2)
