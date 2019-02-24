import numpy as np 
import sklearn
import nltk
import data_loader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from FFN_example import FFN

class bias_classifier():
    def __init__(self, data_path, data_fraction):
        self.dataset  = data_loader.load_data(data_path, data_fraction, random_seed=123456, label_mode=2)
        self.training_data, self.validation_data, self.testing_data, self.training_label, self.validation_label, self.testing_label = self.dataset


    def feature_extraction(self):
        self.count_vector = CountVectorizer(min_df = 10, max_features= 100)
        self.tfidf_transformer = TfidfTransformer()
        train_counts = self.count_vector.fit_transform(self.training_data)
        self.training_data = self.tfidf_transformer.fit_transform(train_counts)

        val_counts = self.count_vector.transform(self.validation_data)
        self.validation_data = self.tfidf_transformer.transform(val_counts)

        test_counts = self.count_vector.transform(self.testing_data)
        self.testing_data = self.tfidf_transformer.transform(test_counts)

        # todo: more ways to extact features

    def reset_training_data(self):
        self.training_data, self.validation_data, self.testing_data, self.training_label, self.validation_label, self.testing_label = self.dataset

    def validation(self, clf):
        predicts = clf.predict(self.validation_data)
        self.report_result(self.validation_label, predicts, 'validation')

    def testing(self, clf):
        predicts = clf.predict(self.testing_data)
        self.report_result(self.testing_label, predicts, 'testing')

    def report_result(self, groundTruth, predicts, name):
        print('-------------------------{name} result-----------------------------'.format(name = name))
        print('accuracy : ', metrics.accuracy_score(groundTruth, predicts))
        print(metrics.classification_report(groundTruth, predicts))
        print('--------------------------end of result-----------------------------')

        # todo: more evaluation metrics or graphs (AUC, ROC, ...)

    def train(self, model_name = 'NB'):
        if model_name == 'NB':
            clf = MultinomialNB()
            clf.fit(self.training_data, self.training_label)
        elif model_name == 'SVM':
            clf = SVC(C=0.1,gamma = 'auto')
            clf.fit(self.training_data, self.training_label)
        elif model_name == 'FFN':
            clf = FFN(num_epochs = 1, batch_size = 100, lr = 0.01, feature_size = 100)

            ###################################################################################
            # due to invalid of features here using dummy input instead need to modify future # 
            ###################################################################################
            clf.fit(np.random.randn(1000,100), np.random.randint(2,size=(1000)), np.random.randn(200,100), np.random.randint(2,size=(200)))
        else:
            raise RuntimeError('No such classifier {name}'.format(name = model_name))
        predicts = clf.predict(self.training_data)
        # predicts = clf.predict(np.random.randn(200,100))
        self.report_result(self.training_label, predicts, 'training')

        return clf
        # todo: more classifiers


    



if __name__ == '__main__':
    c = bias_classifier(['non_bias.pkl', 'bias.pkl', 'moderate_bias.pkl'], [0.7,0.1])
    # c.feature_extraction()
    clf = c.train(model_name='SVM')
    # c.validation(clf)
    # c.testing(clf)


