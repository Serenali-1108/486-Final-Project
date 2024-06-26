# BERT_V2

BERT_V2 is a Python file for training and predicting the attitude toward ChatGPT based on twitter posts by using BERT language model. Including neutral(0), positive(1) and negative(-1). It uses fastbert library and 80% of the provided dataset to train the model and rest of the 20% for testing.

## Download data file, kernel file and trained model
Use this link to download all three large files that are required: https://drive.google.com/drive/folders/1NLi6yQ-WmnRGTtq0Rrwg5nfRklUnytlm

twitter.csv and fastbert.bin should be put in the same directory as bert_v2

## Install fastbert

In order to run bert_v2.py, you will need to install fastbert library first.

```bash
pip install fastbert
```

After that, if you want to train your own model, you should include the kernel file(google_bert_base_en.bin) in the .fastbert directory. Else, no extra modification is required. 

## Command Line Arguments
The first command line argument(twitter.csv) is the data file that you want to use to train the model. The second command line argument(bert_v2.result) is the file that you want to generate for output. 

```bash
python bert_v2.py twitter.csv bert_v2.result
```

## Usage
We have provide our trained model(fastbert.bin). This file should be include in the same directory with bert_v2.py. However, if you want to train your own model, you can uncomment the following code.
```python
    # print("training")
    # model.fit(
    #     contents_train,
    #     labels_train,
    #     model_saving_path='./fastbert.bin',
    # )
```

## Output
The output file include the accuracy, precision, recall and f1 score for the whole prediction and each class(1, 0, -1).

## 486 final project submission only
We have also create a GitHub repo for this project: https://github.com/Serenali-1108/486-Final-Project