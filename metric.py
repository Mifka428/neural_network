import csv
import os

import numpy as np
import pandas as pd
from nltk.translate.meteor_score import single_meteor_score
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from transformers import GPT2Tokenizer

PATH_TO_DATA = 'dataset/train/'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = pd.read_csv(os.path.join(PATH_TO_DATA, "train.csv"))
text_tokens = [tokenizer.encode(description, add_special_tokens=True, truncation=True, max_length=1024) for description
               in dataset['description']]

max_len = max(len(tokens) for tokens in text_tokens)
padded_tokens = pad_sequences(text_tokens, maxlen=max_len + 1, padding='post', truncating='post')


def read_file_list():
    text_list = []
    for i in range(100):
        with open(f'dataset/test/test_stt/{i}.txt', 'r', encoding='utf-8') as f:
            text = ""
            for line in f:
                text += line + " "
            text_list.append(text)
    return text_list


def metric(text_list):
    tokenizer_base = Tokenizer()
    model = load_model("des_gen.h5")

    predictions = model.predict(padded_tokens[:, :-1])
    predicted_classes = np.argmax(predictions, axis=1)
    metric = []
    for i in text_list:
        sequence = tokenizer_base.texts_to_sequences([i])
        data = pad_sequences(sequence, maxlen=100)
        result = model.predict(data)
        summary_text = tokenizer.decode(result.tolist()[0])
        metric.append(single_meteor_score([i], [summary_text]))
    return metric


def save_metric(metric):
    with open('metric.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["video_name", "metric"]
        writer.writerow(field)
        count = 0
        for i in metric:
            writer.writerow([f"{count}.mp4", i])
            count += 1


text_list = read_file_list()
metric_list = metric(text_list)
save_metric(metric_list)
