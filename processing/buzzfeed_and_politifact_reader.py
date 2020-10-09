import processing.reader as reader
import pandas as pd

buzzfeed_real = reader.get_dataframe('BuzzFeed_and_PolitiFact/BuzzFeed_real_news_content.csv')
buzzfeed_real['fake'] = 0
politifact_real = reader.get_dataframe('BuzzFeed_and_PolitiFact/Politifact_real_news_content.csv')
politifact_real['fake'] = 0
buzzfeed_fake = reader.get_dataframe('BuzzFeed_and_PolitiFact/BuzzFeed_fake_news_content.csv')
buzzfeed_fake['fake'] = 1

fake_data = reader.get_dataframe('BuzzFeed_and_PolitiFact/fake.csv')
fake_data = reader.extract_feature(fake_data, language='english', type='fake')
reader.rename_columns(fake_data, source='site_url', authors='author')
fake_data['fake'] = 1

df = pd.concat([reader.extract_columns(buzzfeed_fake), reader.extract_columns(buzzfeed_real),
                reader.extract_columns(politifact_real), reader.extract_columns(fake_data)])

df = reader.preprocess(df, k=10000)
df = reader.extract_columns(df, ('tokenized text', 'tokenized title', 'authors',
                                 'citation_count', 'avg sentence len', 'punctuation count', 'fake'))
reader.write(df, 'buzzfeed_politifact_news.csv')
