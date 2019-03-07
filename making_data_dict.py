import nltk
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
import numpy as np
import codecs
import re
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english')) 

def preprocessing(file):
    global tokenizer, stop_words, stemmer
    dic = {}
    conflict = {}
    tmp_file = open('tmp.txt', 'w')
    for l in file.readlines():
        line = l.strip().split('\t')
        label = line[0]
        sentence = line[1]
        sentence = re.sub('[^A-Za-z.?!]', ' ', sentence)
        tmp_file.write(sentence+'\n')
        sentence = tokenizer.tokenize(sentence)
        sentence = [w.lower() for w in sentence]
        sentence = [w for w in sentence if not w in stop_words]
        # for w in range (len(sentence)):
        #     sentence[w] = stemmer.stem(sentence[w])
        sentence = ' '.join(sentence)
        if sentence in list(dic.keys()):
            print('conflict in : ', sentence)
            print('original label : ', dic[sentence])
            print('new label : ', label)
            if sentence in list(conflict.keys()):
                conflict[sentence] += 1
            else:
                conflict[sentence] = 1
        else:
            dic[sentence] = label
    tmp_file.close()
    return dic, conflict


if __name__ == '__main__':
    bias = codecs.open('./dataset/biased_reviews.txt', 'r', 'utf-8')
    # moderate_bias = codecs.open('./dataset/moderated_biased_reviews.txt', 'r', 'utf-8')
    # non_bias = codecs.open('./dataset/non_biased_reviews.txt', 'r', 'utf-8')

    bias_data, bias_conflict = preprocessing(bias)
    # moderate_bias_data, moderate_bias_conflict = preprocessing(moderate_bias)
    # non_bias_data, non_bias_conflict = preprocessing(non_bias)

    # print(len(bias_data))
    # print(len(moderate_bias_data))
    # print(len(non_bias_data))

    # print(bias_conflict)
    # print(moderate_bias_conflict)
    # print(non_bias_conflict)

    # bias_file = open('bias.pkl', 'wb')
    # moderate_bias_file = open('moderate_bias.pkl', 'wb')
    # non_bias_file = open('non_bias.pkl', 'wb')

    # pickle.dump(bias_data, bias_file)
    # pickle.dump(moderate_bias_data, moderate_bias_file)
    # pickle.dump(non_bias_data, non_bias_file)

    # bias_file.close()
    # moderate_bias_file.close()
    # non_bias_file.close()

