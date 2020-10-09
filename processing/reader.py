from processing.cleaning import keep_top_k_words, apply_all
import pandas as pd
from nltk import FreqDist
import numpy as np
import re

pd.set_option("display.max_columns", 30, 'display.expand_frame_repr', False)


def _avg_sentence_len(text):
    sentence_list = text.split(". ")
    total_len = 0
    for sentence in sentence_list:
        total_len += len(sentence.split(" "))
    return total_len / len(sentence_list)


def _punctuation_count(text, punctuations='!?'):
    sentence_list = text.split(". ")
    count = 0
    for sentence in sentence_list:
        punc_list = [char for char in sentence if char in punctuations]
        count += len(punc_list)
    return count


def _citation_count(text):
    """
        Citation with format "<citation>". e.g: "This is a citation"
    """
    pattern = r'"[^"]*"'
    results = re.finditer(pattern, text)
    citations = [text[match.start(): match.end()] for match in results]
    citations = list(set(citations))
    """
        APA citation: (name, year)
    """
    apa_pattern = r'\(([^"\)]*|\bAnonymous\b|"[^"\)]*")(, )([\d]+|n\.d\.|[\d]+[\w])\)'
    apa_results = re.finditer(apa_pattern, text)
    apa_citations = [text[match.start(): match.end()] for match in apa_results]
    apa_citations = list(set(apa_citations))

    return len(citations) + len(apa_citations)


def _tokenize(df, columns=('text', 'title'), k=None):
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


def extract_columns(df, columns=('title', 'text', 'authors', 'fake')):
    return df[list(columns)]


def extract_feature(df, **kwargs):
    condition = []
    for key, value in kwargs.items():
        condition.append(df[key] == value)
    return df[np.bitwise_and.reduce(condition)]


def get_dataframe(filepath, **kwargs):
    directory = '../data/' + filepath
    df = pd.read_csv(directory, **kwargs)
    return df


def get_text_features(df):
    df['citation_count'] = df.apply(lambda row: _citation_count(row['text']), axis=1)
    df['avg sentence len'] = df.apply(lambda row: _avg_sentence_len(row['text']), axis=1)
    df['punctuation count'] = df.apply(lambda row: _punctuation_count(row['text']), axis=1)


def preprocess(df, columns=('text', 'title'), k=None):
    df.dropna(subset=list(columns), inplace=True)
    get_text_features(df)
    return _tokenize(df, columns, k)


def rename_columns(df, **kwargs):
    columns = {}
    for key, value in kwargs.items():
        columns[value] = key
    df.rename(columns=columns, inplace=True)


def write(df, filename, index=False):
    directory = '../cleaned_data/' + filename
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(directory, index=index)
