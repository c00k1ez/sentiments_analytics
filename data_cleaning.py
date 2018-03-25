import pandas as pd
from pymorphy2 import MorphAnalyzer
import re
import progressbar
from multiprocessing import Pool

TEST = True

positive_tweets = pd.read_csv('data/positive.csv', sep = ';', header = None)
negative_tweets = pd.read_csv('data/negative.csv', sep = ';', header = None)
dataset = pd.concat([positive_tweets, negative_tweets])[[3, 4]]
dataset.columns = ['text', 'label']

dataset = dataset.sample(frac = 1, random_state = 42).reset_index(drop = True)

analyzer = MorphAnalyzer() 

def clean(txt, use_lemmatization = False):
    txt = txt.lower()
    txt = re.sub(r'\s+', ' ', txt)
    letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '
    result = ''
    for letter in txt:
        if letter in letters:
            result += letter
    if use_lemmatization:
        temp = []
        for word in result.split():
            temp.append(analyzer.parse(word)[0].normal_form)
        result = ' '.join(temp)
    return result

if not TEST:
	dataset_with_lemmatization = dataset['text'].apply(lambda x : clean(x, use_lemmatization = True))
	dataset_without_lemmatization = dataset['text'].apply(lambda x : clean(x, use_lemmatization = False))
else:
	pool = Pool()
	dataset_without_lemmatization = dataset['text']
	bar = progressbar.ProgressBar(max_value=len(dataset['text']))
	for i in range(len(dataset['text'])):
		dataset_without_lemmatization[i] = clean(dataset_without_lemmatization[i], use_lemmatization = False)
		bar.update(i)

	dataset_with_lemmatization = dataset['text']
	bar = progressbar.ProgressBar(max_value=len(dataset['text']))
	for i in range(len(dataset['text'])):
		dataset_with_lemmatization[i] = clean(dataset_with_lemmatization[i], use_lemmatization = False)
		bar.update(i)

dataset_with_lemmatization.to_csv("data/data_with_lemmatization.csv")
dataset_without_lemmatization.to_csv("data/data_without_lemmatization.csv")
