import argparse
import re
import pickle
import nltk
import utils

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

spin = utils.Spinner()


def train(raw_data, with_features, raw_dev=None, classi='DT', use_bin=False, use_all_liwc=False, name='', file='', quiet=False):
  if not quiet:
    if name != '':
      name = ' ' + utils.COLORS['blue'] + name + utils.RESET
    if file != '':
      file = ' using ' + utils.BOLD + file + utils.RESET
    add = name + file
    spin.set_strings('Training{0}...'.format(add), 'Trained{0}.'.format(add))
    spin.start()

  label0, label1, label2 = utils.get_reviews(raw_data)

  reviews = [(text, 'non_biased') for text in label0] + \
      [(text, 'moderated_biased') for text in label1] + \
      [(text, 'biased') for text in label2]
  print("with {} data".format(len(reviews)))

  train_data = [((utils.get_features(text, with_features, get_bin=use_bin,
                                     get_all_liwc=use_all_liwc)), label) for text, label in reviews]

  if classi == 'DT':
    classifier = nltk.classify.DecisionTreeClassifier.train(
        train_data, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10)
  elif classi == 'SciDT':
    classifier = SklearnClassifier(DecisionTreeClassifier()).train(
        train_data, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10)
  elif classi == 'NB':
    classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
  elif classi == 'SciNB':
    classifier = SklearnClassifier(BernoulliNB()).train(train_data)
  elif classi == 'SVM':
    classifier = SklearnClassifier(LinearSVC()).train(train_data)
  elif classi == 'LR':
    classifier = SklearnClassifier(LogisticRegression()).train(train_data)

  if not quiet:
    spin.stop()

  return classifier


def classify(classifier, raw_data, with_features, use_bin=False, use_all_liwc=False, name='', file='', quiet=False):
  if not quiet:
    if file != '':
      file = ' ' + utils.BOLD + file + utils.RESET
    if name != '':
      name = ' using ' + utils.COLORS['blue'] + name + utils.RESET
    add = file + name
    spin.set_strings('Clasifying{0}...'.format(
        add), 'Classified{0}.'.format(add))
    spin.start()

  label0, label1, label2 = utils.get_reviews(raw_data)
  # print(len(pos))
  reviews = [(text, 'non_biased') for text in label0] + \
            [(text, 'moderated_biased') for text in label1] + \
            [(text, 'biased') for text in label2]

  class_data = [((utils.get_features(text, with_features, get_bin=use_bin,
                                     get_all_liwc=use_all_liwc)), label) for text, label in reviews]

  acc = nltk.classify.accuracy(classifier, class_data)

  if not quiet:
    spin.set_strings(
        se='Classified{0}. Accuracy: {1:.2f}%'.format(add, acc * 100))
    spin.stop()

  return acc, classifier.classify_many([feat for feat, label in class_data]), [label for feat, label in class_data]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Fake Reviews Baseline')
  parser.add_argument('-t', '--train', dest='is_training', default=False,
                      help='Set to train new model.', action='store_true')
  parser.add_argument('-d', '--data', dest='data', default=None, nargs='+',
                      help='Data file to use for training/classification.', type=argparse.FileType('rb'))

  parser.add_argument('-c', '--classifier', dest='classi',
                      default='DT', help='Type of classifier to use. One of {"DT", "NB", "SciDT", "SciNB", "SVM", "LR"}')
  parser.add_argument('-s', '--tmodel', dest='tmodel',
                      help='Model (.pickle file) to use to train.', type=argparse.FileType('wb'))
  parser.add_argument('-m', '--model', dest='model',
                      help='Model (.pickle file) to use to classify.', type=argparse.FileType('rb'))

  parser.add_argument('-o', '--output', dest='out', default=None,
                      help='Output filename (.pickle or .txt)', type=argparse.FileType('w'))

  parser.add_argument('-f', '--features', dest='feat', default=['all'], nargs='*',
                      help='Feature sets to use. List with elements from {"word", "pos", "liwc", "w2v", "all"}')
  parser.add_argument('-b', '--bin', dest='binning', default=False,
                      help='Use binning (word and pos).', action='store_true')
  parser.add_argument('-a', '--allliwc', dest='allliwc', default=False,
                      help='Use all liwc (liwc).', action='store_true')

  parser.add_argument('-q', '--quiet', dest='quiet', default=False,
                      help='Quiet mode (no console output).', action='store_true')

  try:
    args = parser.parse_args()

    raw_data = args.data[0].read().decode("latin1")

    if args.is_training:
      if len(args.data) > 1:
        raw_dev = args.data[1].readlines()
      else:
        raw_dev = None

      classifier = train(raw_data, args.feat, raw_dev=raw_dev, use_bin=args.binning, use_all_liwc=args.allliwc,
                         classi=args.classi, name=args.tmodel.name, file=args.data[0].name, quiet=args.quiet)
      pickle.dump(classifier, args.tmodel)

    else:
      classifier = pickle.load(args.model)
      acc, res, ref = classify(classifier, raw_data, args.feat, use_bin=args.binning,
                               use_all_liwc=args.allliwc, name=args.model.name, file=args.data[0].name, quiet=args.quiet)

      if args.out is not None:
        args.out.write('# Accuracy: {:.2f}%\n\n'.format(acc * 100))
        cm = nltk.ConfusionMatrix(ref, res)
        args.out.write('# Confusion Matrix:\n{}'.format(cm.pretty_format()))
        if not args.quiet:
          print('    Results saved to {}.'.format(args.out.name))

  except IOError as msg:
    parser.error(str(msg))
