library(tidyverse)
library(dplyr)
library(stringr)
library(broom)
library(gridExtra)
library(tm)
library(SnowballC)


train <- read.csv("train_new.csv")

str(train)

View(head(train))

#fnd_news <- read.csv("fnd_news_compiled.csv")

View(head(train))

clean_dataset_one_text_column <- function(train){
  train <- apply(train, 2, str_to_lower)
  
  train[,1] <- removeNumbers(train[,1])
  
  train[,1] <- removeWords(train[,1], stopwords("english"))
  
  train[,1] <- removePunctuation(train[,1])
  
  train[,1] <- stemDocument(train[,1], language = "english")
  
  train[,1] <- str_replace_all(train[,1], "[^[:alnum:]]", " ")
  
  train[,1] <- gsub('[^\x20-\x7E]', '', train[,1])
  
  train[,1] <- str_squish(train[,1])
  
  return (train)
}

train_clean <- clean_dataset_one_text_column(train)
View(head(train_clean))

######

fnd_news_tokenised <- read.csv("fnd_news_compiled_tokenised - update.csv")
View(head(fnd_news_tokenised))

fnd_to_use <- select(fnd_news_tokenised, text, is_fake)

fnd_to_use <- rename(fnd_to_use, label = is_fake)

View(head(fnd_to_use))
#######

buzz_fake <- read.csv("BuzzFeed_fake_news_content.csv", stringsAsFactors = FALSE)

buzz_fake_trim <- select(buzz_fake, title, text)

buzz_fake_trim <- mutate(buzz_fake_trim, label = 1)

buzz_real <- read.csv("BuzzFeed_real_news_content.csv", stringsAsFactors = FALSE)

buzz_real_trim <- select(buzz_real, title, text)
buzz_real_trim <- mutate(buzz_real_trim, label = 0)

buzz_news <- rbind(buzz_fake_trim, buzz_real_trim)

View(head(buzz_news))

buzz_news_to_use <- select(buzz_news, text, label)

buzz_news_clean <- clean_dataset_one_text_column(buzz_news_to_use)

View(head(buzz_news_clean))

#########

buzz_thanh <- read.csv("buzzfeed_news_thanh.csv")
View(head(buzz_thanh))

buzz_thanh_new <- select(buzz_thanh, tokenized.text, fake)

buzz_thanh_to_use <- rename(buzz_thanh_new, text = tokenized.text, label = fake)

buzz_thanh_to_use_clean <- clean_dataset_one_text_column(buzz_thanh_to_use)

View(head(buzz_thanh_to_use_clean))

###########

complete_dataset <- rbind(train_clean, fnd_to_use, buzz_news_clean, buzz_thanh_to_use_clean)
View((complete_dataset[130:140,]))

dim(complete_dataset)

dim(filter(complete_dataset, label == 1)) # 3394 fake, 2436 real

write.csv(complete_dataset,'complete_news_dataset_290920.csv')
