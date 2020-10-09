import processing.reader as reader
import pandas as pd

real_df = reader.get_dataframe('fakeandrealnewsdataset/True.csv')
real_df['fake'] = 0
real_df = reader.extract_columns(real_df, columns=('title', 'text', 'fake'))
fake_df = reader.get_dataframe('fakeandrealnewsdataset/Fake.csv')
fake_df['fake'] = 1
fake_df = reader.extract_columns(fake_df, columns=('title', 'text', 'fake'))

df = pd.concat([real_df, fake_df])

df = reader.preprocess(df, k=10000)
df = reader.extract_columns(df, ('tokenized text', 'tokenized title', 'citation_count',
                                 'avg sentence len', 'punctuation count', 'fake'))
reader.write(df, 'fake_and_real_news.csv')
