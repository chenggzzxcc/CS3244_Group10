import processing.reader as reader

df = reader.get_dataframe('FA_KES/FA-KES-Dataset.csv', encoding='unicode_escape', engine='python')
reader.rename_columns(df, title='article_title', text='article_content', fake='labels')
df = reader.extract_columns(df, columns=('title', 'text', 'fake'))
df['fake'] = 1 - df['fake']

df = reader.preprocess(df, k=10000)
df = reader.extract_columns(df, ('tokenized text', 'tokenized title', 'citation_count',
                                 'avg sentence len', 'punctuation count', 'fake'))
reader.write(df, 'fakes_news.csv')
