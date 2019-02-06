--------------------------CMPS290 project-----------------------------------------
#
# Fake Reviews Baseline
# Jinrui Yang jyang193@ucsc.edu
# Feb 6th 2019
#
-------------------------------------------------------------------------------------
Hi there. This baseline program includes four models: naive bayes, decision tree, svm and logistic regression, include four features: word, pos, liwc, and word embedding(using Google's w2v model).

Gonna start explaining how baseline program add arguments:

$ python3 baseline.py -d DATA_FILE -c CLASSIFIER [-s OUTPUT_PICKLE]
                    [-m INPUT_PICKLE] [-o OUTPUT_RESULT] -f [FEAT 1 [FEAT 2 ...]]
                    [-b] [-a] [-t] [-q]

  -d lets specify the source of data. such as
      data/train_examples.txt.

  -c lets select the type of classifier from {DT, NB, SciDT, SciNB, SVM, LR}.

  -s specifies the file to save a training classifier.

  -m specifies the file to load a classifier from.

  -o sets the file where classifying results will be saved to.

  -f lets select which features to use from {word, pos, liwc, w2v}. More than one can
      be specify or 'all' to use all of them.

  -b sets using binning to true.

  -a sets using all LIWC featues to true.

  -t sets training mode.

  -q sets quiet mode (no output in console).
-------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------

This runs each of the model. Spots to review:

# Train model:

Let using word and pos features to train decission tress model as an example:

$ python3 baseline.py -d data/train_examples.txt
                      -c DT
                      -s classifiers/dt-word_pos_features.classifier.pickle
                      -f word pos
                      -t

# Validate model:
$ python3 baseline.py -d data/dev_examples.txt
                      -m classifiers/dt-word_pos_features.classifier.pickle
                      -f word pos
                      -o results/word_pos_features_DT_dev_results.txt

# Test model:
$ python3 baseline.py -d data/test_examples.txt
                      -m classifiers/dt-word_pos_features.classifier.pickle
                      -f word pos
                      -o results/word_pos_features_DT_test_results.txt


-------------------------------------------------------------------------------------

NOTE1: Google's w2v model is expected to be in a /data directory. This path can be
       changed in utils.py (top of the file).

NOTE2: LIWC files are expected to be in a data/ directory.

NOTE3: The train/development/test data in this floder are not complete, they are subset
       of the raw data.

-------------------------------------------------------------------------------------
Included these files in the BASELINE folder:


  - utils.py
  - baseline.py
  - word_category_counter.py
  - word2vec_extractor.py


  - data/train_examples.txt
  - data/dev_examples.txt
  - data/test_examples.txt
  - data/liwc/LIWC2007.dic
  - data/liwc/DefaultDic2003.dic
  - data/liwc/_DS_Store


  - classifiers/word-DT-classifier.pickle
  - classifiers/word-LR-classifier.pickle

  - results/word_features_DT_dev_results.txt
  - results/word_features_LR_dev_results.txt

  - README.txt
