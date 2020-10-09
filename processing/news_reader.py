import processing.reader as reader

news_df = reader.get_dataframe('fake_news/train.csv')
reader.rename_columns(news_df, fake='label', authors='author')
news_df = reader.extract_columns(news_df)

df = reader.preprocess(news_df, k=10000)
df = reader.extract_columns(df, ('tokenized text', 'tokenized title', 'authors',
                                 'citation_count', 'avg sentence len', 'punctuation count', 'fake'))
reader.write(df, 'news.csv')
