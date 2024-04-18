# 486 Final Project
Download data from https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023/data, name it `twitter.csv`.

## Part 1. ML and DL

**Step 0. Data Labeling**

1. Run `textblob.py`, which use TextBlob to label and tokenize each Twitter post. Polarity of each post is encoded as categorical labels ([-1,0) = negative, 0 = neutral, (0,1] = postitive)

3. Run tuning_script `sample.py` to obtain a 1% stratefied sample, 0.01_twitter.csv (used for parameter tuning later)

**Step 1. Data Preprocessing**

Run `vectorize.py` to get TF-IDF of each post with feature extraction. 

(Optional: parameter tuning)
Run `explore_tfidf.py` to tune parameters for TF-IDF and SVD

(a). Vectorizing data using TF-IDF 
Use sampled dataset(0.01_twitter.csv) to choose min_df and max_df parameters for tf-idf vectorizer. It does the following: 
1. Remove stopwords
2. Use grid search to explore values of min_df, max_df, choose the parameters that results in highest accuracy when fitting a simple logistic regression model.

(b). Dimensionality reduction using Singular Value Decomposition (SVD). It does the following: 
1. Use grid search on sampled dataset(0.01_twitter.csv) to explore num_components
2. Train a simple logistic regression model using each num_component, cross validating with 5-folds
3. Plot cross-validated accuracy and select optimal num_components

**Step 2. Hyperparameter Tuning** 

*1. Logistic Regression*:

Run tuning_script `sample_log_reg.py` to use grid search and cross validation on sampled dataset (0.01_twitter.csv) to find best parameters for logistic regression. Best parameters are printed.

*2. SVM*:

Run tuning_script `sample_svm.py` to use grid search and cross validation on sampled dataset (0.01_twitter.csv) to find best parameters for SVM. Best parameters are printed.

**Step 3. Model training and testing**

*1. Logistic Regression*:

Run `logistic_reg.py` using best parameters from step 2 to train a LR classifier on 80% of entire data and test on 20%. Classification report is printed and prediction output saved to `log_reg_output.csv`.

*2. LinearSVM*:

Run `linear_svm.py` using best parameters from step 2 to train a linear SVM classifier on 80% of entire data and test on 20%. Classification report is printed and prediction output saved to `linear_svm_output.csv`.

*3. Neural Network*:

i. Run `nn_train.py` to train a neural network on 80% of entire data.
ii. Run `nn_test.py` to test the neural network on 20% of entire data. Classification report is printed and prediction output saved to `nn_output.csv`.

**Step 4. Create prediction plots**

Run `plot.py` to obtain confution matrix using  `log_reg_output.csv`, `linear_svm_output.csv` or `nn_output.csv`. (change csv file name to be read in)

## Part 2. FastBERT
﻿# BERT_V2

BERT_V2 is a Python file for training and predicting the attitude toward ChatGPT based on twitter posts by using BERT language model. Including neutral(0), positive(1) and negative(-1). It uses fastbert library and 80% of the provided dataset to train the model and rest of the 20% for testing.

**Step 0. Download data file, kernel file and trained model**
Use this link to download all three large files that are required: https://drive.google.com/drive/folders/1NLi6yQ-WmnRGTtq0Rrwg5nfRklUnytlm

twitter.csv and fastbert.bin should be put in the same directory as bert_v2

**Step 1. Install fastbert**

In order to run bert_v2.py, you will need to install fastbert library first.

```bash
pip install fastbert
```

After that, if you want to train your own model, you should include the kernel file(google_bert_base_en.bin) in the .fastbert directory. Else, no extra modification is required. 

**Step 2. Command Line Arguments**
The first command line argument(twitter.csv) is the data file that you want to use to train the model. The second command line argument(bert_v2.result) is the file that you want to generate for output. 

```bash
python bert_v2.py twitter.csv bert_v2.result
```

**Step 3. Training Model**
We have provide our trained model(fastbert.bin). This file should be include in the same directory with bert_v2.py. However, if you want to train your own model, you can uncomment the following code.
```python
    # print("training")
    # model.fit(
    #     contents_train,
    #     labels_train,
    #     model_saving_path='./fastbert.bin',
    # )
```

** Output **
The output file include the accuracy, precision, recall and f1 score for the whole prediction and each class(1, 0, -1).

