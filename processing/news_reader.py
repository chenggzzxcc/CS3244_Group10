import processing.reader as reader

news_df = reader.get_dataframe('fake_news/train.csv')
reader.rename_columns(news_df, fake='label', authors='author')
news_df = reader.extract_columns(news_df)

df = reader.tokenize_columns(news_df, k=10000)
df = reader.extract_columns(df, ('tokenized text', 'tokenized title', 'authors', 'fake'))
reader.write(df, 'news.csv')
