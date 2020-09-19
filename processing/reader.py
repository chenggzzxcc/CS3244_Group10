from processing.cleaning import keep_top_k_words, apply_all
import pandas as pd
from nltk import FreqDist
import numpy as np

pd.set_option("display.max_columns", 30, 'display.expand_frame_repr', False)


def get_dataframe(filepath, **kwargs):
    directory = '../data/' + filepath
    df = pd.read_csv(directory, **kwargs)
    return df


def rename_columns(df, **kwargs):
    columns = {}
    for key, value in kwargs.items():
        columns[value] = key
    df.rename(columns=columns, inplace=True)


def extract_columns(df, columns=('title', 'text', 'authors', 'fake')):
    return df[list(columns)]


def extract_feature(df, **kwargs):
    condition = []
    for key, value in kwargs.items():
        condition.append(df[key] == value)
    return df[np.bitwise_and.reduce(condition)]


def tokenize_columns(df, columns=('text', 'title'), k=None):
    notnull_condition = np.array([df[column].notnull() for column in columns])
    df = df[np.bitwise_and.reduce(notnull_condition)]
    df.drop_duplicates()
    tokenized_columns = ['tokenized ' + column for column in columns]
    df2 = df.copy(deep=True)
    for i in range(len(columns)):
        df2[tokenized_columns[i]] = df2[columns[i]].apply(apply_all)
    if k is not None:
        tokenized_list = np.array([df2[tokenized_column] for tokenized_column in tokenized_columns])
        all_words = [word for item in list(np.add.reduce(tokenized_list)) for word in item]
        fdist = FreqDist(all_words)
        top_k_words, _ = zip(*fdist.most_common(k))
        top_k_words = set(top_k_words)
    for tokenized_column in tokenized_columns:
        df2[tokenized_column] = df2[tokenized_column].apply(lambda x: keep_top_k_words(x, top_k_words))
        df2[tokenized_column] = df2[tokenized_column].apply(lambda text: ' '.join(text))
    return df2


def write(df, filename, index=False):
    directory = '../cleaned_data/' + filename
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(directory, index=index)



# news_df = pd.read_csv('../../data/fake_news/train.csv')
# news_df.rename(columns={'author': 'authors', 'label': 'fake'}, inplace=True)
# df = news_df[['title', 'text', 'authors', 'fake']]
# df = df[(df['text'].notnull()) & (df['title'].notnull())]

# df.drop_duplicates()
#
# df['tokenized text'] = df['text'].apply(apply_all)
# df['tokenized title'] = df['title'].apply(apply_all)
#
# k = 10000
# all_words = [word for item in list(df['tokenized text'] + df['tokenized title']) for word in item]
# fdist = FreqDist(all_words)
# top_k_words, _ = zip(*fdist.most_common(k))
# top_k_words = set(top_k_words)
# df['tokenized text'] = df['tokenized text'].apply(lambda x: keep_top_k_words(x, top_k_words))
# df['tokenized text'] = df['tokenized text'].apply(lambda text: ' '.join(text))
# df['tokenized title'] = df['tokenized title'].apply(lambda text: ' '.join(text))
# cleaned_df = df[['tokenized text', 'tokenized title', 'authors', 'fake']]
# cleaned_df = cleaned_df.sample(frac=1).reset_index(drop=True)
# cleaned_df.to_csv('../data/extracted/news.csv', index=False)


