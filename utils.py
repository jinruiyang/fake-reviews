import os
import sys
import nltk
import re
import pickle
import itertools
import threading
import time
import word_category_counter
from word2vec_extractor import Word2vecExtractor
import pandas as pd
import csv

w2v_path = 'GoogleNews-vectors-negative300.bin'

RESET = '\u001b[0m'
COLORS = {'red': '\u001b[31m', 'green': '\u001b[32m',
    'yellow': '\u001b[33m', 'blue': '\u001b[34m'}
BOLD = '\u001b[1m'


def log(s, color=None, bold=False):
  c = ''
  if color is not None and color in COLORS:
    c = COLORS[color]

  if bold:
    c = c + BOLD

  print(c + s + RESET)

def save_classifier(name, object):
  with open(name, 'wb') as f:
    pickle.dump(object, f)
    f.close()
  return name

def load_classifier(filename):
  with open(filename, 'rb') as f:
    c = pickle.load(f)
    f.close()
  return c


def read_file(fname):
  with open(fname, "rb") as fin:
    raw_data = fin.read().decode("latin1")
  return raw_data


def get_label(review):
  return int(review[0])


def get_text(review):
  return review[2:]


def get_reviews(raw_data):
  label0_reviews = []
  label1_reviews = []
  label2_reviews = []
  first_sent = None
  for review in re.split(r'\n', raw_data):
    if review:
        if get_label(review) == 0:
         label0_reviews.append(review[2:])
        if get_label(review) == 1:
         label1_reviews.append(review[2:])
        if get_label(review) == 2:
         label2_reviews.append(review[2:])

  return label0_reviews,label1_reviews,label2_reviews


def bin(count):
  return count if count < 2 else 3


def normalize(token, should_normalize=True):
  stopwords = nltk.corpus.stopwords.words('english')

  if not should_normalize:
    normalized_token = token

  else:
    normalized_token = token.lower()
    if normalized_token in stopwords or re.match(r'^(?=\w+).+$', normalized_token) is None:
      normalized_token = None

  return normalized_token


def get_tokens_tags(text, should_normalize=True):
  tokens = []
  tags = []

  # Token sentences, then words per sentence, then normalize
  token_sent = nltk.sent_tokenize(text)
  token_word = [nltk.word_tokenize(sent) for sent in token_sent]
  norma_word = [[normalize(w, should_normalize) for w in sent if normalize(
      w, should_normalize) is not None] for sent in token_word]

  # For word, tag in each sent pos_tagged, add them to the lists
  for sent in norma_word:
    tagging = nltk.pos_tag(sent)

    tokens.append([w for w, t in tagging])
    tags.append([t for w, t in tagging])

  return tokens, tags


def get_pos_features(text, selected_features=None, get_bin=False):
  feature_vectors = {}
  tokens, tags = get_tokens_tags(text, should_normalize=False)

  sents = [' '.join([w for w in sent]) for sent in tags]
  full_text = ' '.join(sents)
  bigrams = []
  for s in tags:
    bigrams = bigrams + list(nltk.bigrams(s))

  dist_uni = nltk.FreqDist(full_text.split(' '))
  dist_big = nltk.ConditionalFreqDist(bigrams)

  for item, freq in dist_uni.items():
    if get_bin == True:
      field = 'UNI_{0}_{1}'.format(item, bin(freq))
      if selected_features is None or field in selected_features:
        feature_vectors[field] = 1
    else:
      field = 'UNI_{0}'.format(item)
      if selected_features is None or field in selected_features:
        feature_vectors[field] = 1

  # for cond in dist_big.conditions():
  #   for item, freq in dist_big[cond].items():
  #     field = 'BIGRAM_{0}_{1}'.format(cond, item)
  #     feature_vectors[field] = 1

  return feature_vectors

def get_lexical_features(text, selected_features=None, get_bin=False):
  feature_vectors = {}
  tokens, tags = get_tokens_tags(text)

  sents = [' '.join([w for w in sent]) for sent in tokens]
  full_text = ' '.join(sents)
  bigrams = []
  for s in tokens:
    bigrams = bigrams + list(nltk.bigrams(s))

  dist_uni = nltk.FreqDist(full_text.split(' '))
  dist_big = nltk.ConditionalFreqDist(bigrams)

  for item, freq in dist_uni.items():
    if get_bin == True:
      field = 'UNI_{0}_{1}'.format(item, bin(freq))
      if selected_features is None or field in selected_features:
        feature_vectors[field] = 1
    else:
      field = 'UNI_{0}'.format(item)
      if selected_features is None or field in selected_features:
        feature_vectors[field] = 1

  # for cond in dist_big.conditions():
  #   for item, freq in dist_big[cond].items():
  #     field = 'BIGRAM_{0}_{1}'.format(cond, item)
  #     feature_vectors[field] = 1

  return feature_vectors


def get_liwc_features(text, selected_features=None, get_all=False):
  feature_vectors = {}
  tokens, tags = get_tokens_tags(text, should_normalize=False)

  text = " ".join([' '.join([w for w in sent]) for sent in tokens])
  liwc_scores = word_category_counter.score_text(text, raw_counts=True)

  if get_all == True:
    return liwc_scores

  negative_score = liwc_scores["Negative Emotion"]
  positive_score = liwc_scores["Positive Emotion"]
  feature_vectors["liwc:neg_emotion"] = negative_score
  feature_vectors["liwc:pos_emotion"] = positive_score

  if positive_score > negative_score:
    feature_vectors["liwc:positive"] = 1
    feature_vectors["liwc:negative"] = 0
  elif positive_score < negative_score:
    feature_vectors["liwc:positive"] = 0
    feature_vectors["liwc:negative"] = 1

  feature_vectors["liwc:swear_words"] = liwc_scores["Swear Words"]
  feature_vectors["liwc:anger"] = liwc_scores["Anger"]
  feature_vectors["liwc:health"] = liwc_scores["Health"]
  feature_vectors["liwc:money"] = liwc_scores["Money"]
  feature_vectors["liwc:pos_feelings"] = liwc_scores["Positive feelings"]
  feature_vectors["liwc:time"] = liwc_scores["Time"]

  return feature_vectors



def get_w2v_features(text):
  w2v_extractor = Word2vecExtractor(w2v_path)
  return w2v_extractor.get_doc2vec_feature_dict(text)


def get_features(text, feat_set=['all'], selected_features=None, get_bin=False, get_all_liwc=False):
  feature_vectors = {}

  if 'all' in feat_set:
    feature_vectors.update(get_lexical_features(text, selected_features, get_bin))
    feature_vectors.update(get_pos_features(text, get_bin))
    feature_vectors.update(get_liwc_features(text, selected_features, get_all_liwc))
    feature_vectors.update(get_w2v_features(text, selected_features))
  else:
    if 'word' in feat_set:
      feature_vectors.update(get_lexical_features(text, selected_features, get_bin))
    if 'pos' in feat_set:
      feature_vectors.update(get_pos_features(text, selected_features, get_bin))
    if 'liwc' in feat_set:
      feature_vectors.update(get_liwc_features(text, selected_features, get_all_liwc))
    if 'w2v' in feat_set:
      feature_vectors.update(get_liwc_features(text, selected_features, get_all_liwc))

  return feature_vectors


class Spinner:
  busy = False
  delay = 0.1

  @staticmethod
  def spinning_cursor():
    while 1:
      for cursor in '|/-\\':
        yield '[' + cursor + ']'

  def __init__(self, string='', endString=None, delay=None):
    self.spinner_generator = self.spinning_cursor()
    self.string = string
    self.end = endString
    if delay and float(delay):
      self.delay = delay

  def set_strings(self, s=None, se=None):
    if s is not None:
      self.string = s
    if se is not None:
      self.end = se

  def spinner_task(self):
    while self.busy:
      sys.stdout.write(COLORS['yellow'] + next(self.spinner_generator) + RESET + ' ' + self.string)
      sys.stdout.flush()
      time.sleep(self.delay)
      sys.stdout.write('{}'.format('\b' * (len(self.string) + 4)))
      sys.stdout.flush()

  def start(self):
    self.busy = True
    threading.Thread(target=self.spinner_task).start()

  def pause(self):
    self.busy = False
    time.sleep(self.delay)

  def stop(self):
    self.busy = False
    time.sleep(self.delay)
    sys.stdout.write('{}\r'.format(' ' * (len(self.string) + 4)))
    if self.end is None:
      sys.stdout.write(COLORS['green'] + '[\u2713]' + RESET + ' ' + self.string + '\n')
    else:
      sys.stdout.write(COLORS['green'] + '[\u2713]' + RESET + ' ' + self.end + '\n')
    # sys.stdout.flush()

