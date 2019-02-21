import nltk
import pickle
import numpy as np
import codecs
import random

def load_data(file_lists, fractions = [0.7, 0.1], random_seed = 123456, label_mode = 1):
    if label_mode > 2:
        raise RuntimeError('Do not have label mode {mode}'.format(mode = label_mode))
    if len(fractions) != 2:
        raise RuntimeError('fractions error')
    if np.sum(fractions) >= 1:
        raise RuntimeError('fractions error')
    fractions[1] = fractions[0] + fractions[1]
    data_set = []
    labels = []
    for f in file_lists:
        file_in = open(f, 'rb')
        data = pickle.load(file_in)
        file_in.close()
        c = 0
        for review in data.keys():
            label = data[review]
            # review = review.split(' ')
            data_set.append(review)
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
    validation_set = data_set[int(len(data_set)*fractions[0]):int(len(data_set)*fractions[1])]
    validation_labels = labels[int(len(data_set)*fractions[0]):int(len(data_set)*fractions[1])]
    testing_set = data_set[int(len(data_set)*fractions[1]):]
    testing_labels = labels[int(len(data_set)*fractions[1]):]

    assert len(training_set) == len(training_labels) and len(validation_set) == len(validation_labels) and len(testing_set) == len(testing_labels)

    print('---------------------summary of dataset-------------------')
    print('{num_train} reviews of training data'.format(num_train = len(training_set)))
    print('{num_val} reviews of validation data'.format(num_val = len(validation_set)))
    print('{num_test} reviews of testing data'.format(num_test = len(testing_set)))
    print('---------------------end of summary-------------------')

    return training_set, validation_set, testing_set, training_labels, validation_labels, testing_labels


if __name__ == '__main__':
    a = load_data(['non_bias.pkl', 'bias.pkl', 'moderate_bias.pkl'], [0.7,0.1],label_mode=2)
