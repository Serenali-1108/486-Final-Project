from fastbert import FastBERT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import sys

def prepare_data(data_file):
    print("prepare_data")
    data = pd.read_csv(data_file)
    contents = data['content'].values
    labels = data['label'].values
    return contents, labels


def train_and_test_bert(contents, labels, output_file):
    contents = [str(item) for item in contents]
    # 1 means positive, 0 means neutral, -1 means negative
    unique_labels = [1, 0, -1]
    out_put_file_name = output_file
    print("google_bert_base_en")

    model = FastBERT(
        kernel_name="google_bert_base_en",
        labels=unique_labels,
        device='cuda'
    )

    # Split data into training and testing sets (80% train, 20% test)
    contents_train, contents_test, labels_train, labels_test = \
        train_test_split(contents, labels, test_size=0.20, random_state=42)

    # uncomment when training is required
    # print("training")
    # model.fit(
    #     contents_train,
    #     labels_train,
    #     model_saving_path='./fastbert.bin',
    # )

    print("loading")
    model.load_model('./fastbert.bin')

    print("testing")
    results = [model(text, speed=0.7) for text in contents_test]
    predictions = [result[0] for result in results]

    # Calculate overall metrics
    accuracy = accuracy_score(labels_test, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(labels_test, predictions, average='weighted')

    with open(out_put_file_name, 'a') as file:
        file.write(f"Overall Accuracy: {accuracy}:\n")
        file.write(f"Overall Precision: {precision}\n")
        file.write(f"Overall Recall: {recall}\n")
        file.write(f"Overall F1-Score: {fscore}\n")
        file.write("\n")
    accuracy_dict = {}

    # Calculate metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels_test, predictions, average=None, labels=unique_labels)

    # Calculate accuracy for each class
    pos_predict_count = 0
    neg_predict_count = 0
    neutral_predict_count = 0

    pos_predict_correct = 0
    neg_predict_correct = 0
    neutral_predict_correct = 0

    for index1 in range(len(predictions)):
        if predictions[index1] == 0:
            neutral_predict_count += 1
            if labels_test[index1] == 0:
                neutral_predict_correct += 1
        elif labels_test[index1] == 1:
            pos_predict_count += 1
            if labels_test[index1] == 1:
                pos_predict_correct += 1
        else:
            neg_predict_count += 1
            if labels_test[index1] == -1:
                neg_predict_correct += 1

    accuracy_dict[1] = pos_predict_correct / pos_predict_count
    accuracy_dict[0] = neutral_predict_correct / neutral_predict_count
    accuracy_dict[-1] = neg_predict_correct / neg_predict_count

    # Output each class's metrics
    with open(out_put_file_name, 'a') as file:
        for i, label in enumerate(unique_labels):
            file.write(f"Metrics for label {label}:\n")
            file.write(f"Accuracy: {accuracy_dict[label]}\n")
            file.write(f"Precision: {precision[i]}\n")
            file.write(f"Recall: {recall[i]}\n")
            file.write(f"F1-Score: {fscore[i]}\n")
            file.write("\n")
    return


def main():
    data_file = sys.argv[1]
    output_file = sys.argv[2]
    contents, labels = prepare_data(data_file)
    train_and_test_bert(contents, labels, output_file)


if __name__ == "__main__":
    main()
