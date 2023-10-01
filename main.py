import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from transformers import GPT2Tokenizer

import pandas as pd
from tqdm import tqdm

from nltk.translate import meteor
from nltk import word_tokenize

from sumy.parsers.plaintext import PlaintextParser

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from nltk.corpus import stopwords
import numpy as np

PATH_TO_DATA = 'dataset/train/'
dataset = pd.read_csv(os.path.join(PATH_TO_DATA, "train.csv"))
dataset.head(5)

with open("dataset/stop-words-russian.txt", 'r', encoding='utf-8') as f:
    extra_stop_words = f.readlines()
    extra_stop_words = [line.strip() for line in extra_stop_words]


def sumy_method(text, n_sent: int = 4):
    parser = PlaintextParser.from_string(text, Tokenizer("russian"))

    stemmer = Stemmer("russian")
    summarizer = LexRankSummarizer(stemmer)
    stopwords_ru = stopwords.words('russian')
    stopwords_ru.extend(extra_stop_words)
    summarizer.stop_words = stopwords_ru

    # Summarize the document with n_sent sentences
    summary = summarizer(parser.document, n_sent)
    dp = []
    if len(summary) > 0:
        for i in summary:
            lp = str(i)
            dp.append(lp)

        final_sentence = ' '.join(dp)
    else:
        final_sentence = ''
    if len(final_sentence.split(" ")) > 512:
        final_sentence = " ".join(final_sentence.split(" ")[:512])
    return final_sentence


tqdm.pandas()


def del_timestamps(text):
    text = text.split("]  ")[1:]
    return " ".join(text)


def gen_description(stt_name, n_sent, category_name):
    with open(os.path.join(PATH_TO_DATA, 'train_stt', stt_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [del_timestamps(line.strip()) for line in lines]
        lines = " ".join(lines)
        res = sumy_method(lines, n_sent)
        if len(res) > 0:
            return res
        else:
            return category_name


def func(stt_name, text, text_sum):
    if isinstance(text_sum, str):
        return round(meteor([word_tokenize(text)], word_tokenize(text_sum)), 4)
    else:
        return 0


# Загрузка предобученного GPT-2 токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Токенизация текста
text_tokens = [tokenizer.encode(description, add_special_tokens=True, truncation=True, max_length=1024) for description
               in dataset['description']]

max_len = max(len(tokens) for tokens in text_tokens)
padded_tokens = pad_sequences(text_tokens, maxlen=max_len + 1, padding='post', truncating='post')

vocab_size = tokenizer.vocab_size
embedding_dim = 768  # размер эмбеддинга для GPT-2

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(512, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

labels = padded_tokens[:, -1:]

model.fit(padded_tokens[:, :-1], labels, epochs=5)


def generate_description(seed_text, max_length=100):
    input_sequence = tokenizer.encode(seed_text, add_special_tokens=True)
    input_sequence = pad_sequences([input_sequence], maxlen=max_len - 1, padding='post', truncating='post')
    generated_sequence = []
    for _ in range(max_length):
        predicted_word_index = np.argmax(model.predict(input_sequence), axis=-1)
        input_sequence = np.concatenate([input_sequence, predicted_word_index], axis=-1)
        generated_sequence.append(predicted_word_index[0, -1])

    generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    return generated_text


# Обучение модели на подготовленных данных
text = """
В предусмотренной планом "Зет" Северо-Кавказской зоне режима повышенной безопасности, центр которой располагался в Тиходонске, продолжали проводиться заградительные, фильтрационные и оперативно-профилактические мероприятия.
На дальних границах режимного периметра бойцы дивизии внутренних войск перекрыли грунтовые сельские дороги и тропинки, которыми пользовались похитители скота, перевозчики наркотиков и прочий малоорганизованный криминальный люд из сопредельных кавказских республик.
На основной трассе стоял суровый тиходонский ОМОН, с которым безуспешно пытались договориться "по-хорошему" представители крупных преступных кланов, привыкших свободно гонять через "прозрачную" границу грузовики с самодельной водкой, оружием, мешками настоящих и фальшивых денег, цистерны с бензином и соляркой.
Несколько раз мощные КамАЗы на скорости прорывались сквозь заслон, снося хлипкие барьеры передвижных ограждений, но омоновцы открывали огонь, дырявили скаты и кузова, а непонятливых водителей вразумляли самым убедительным, хотя и абсолютно незаконным способом.
"""
model.save('des_gen.h5')
tokenizer_base = Tokenizer()
sequence = tokenizer_base.texts_to_sequences([text])
predictions = model.predict(padded_tokens[:, :-1])
predicted_classes = np.argmax(predictions, axis=1)
data = pad_sequences(sequence, maxlen=100)
result = model.predict(data)
summary_text = tokenizer.decode(result.tolist()[0])

print(summary_text)
