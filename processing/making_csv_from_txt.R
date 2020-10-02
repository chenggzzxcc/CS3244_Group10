library(tidyverse)
library(dplyr)
library(stringr)
library(readxl)
library(tm)
library(SnowballC)

fnamesfake <- list.files("fnd_news_fake", pattern="*.txt$", full.names = TRUE)
fnamesreal <- list.files("fnd_news_real", pattern="*.txt$", full.names = TRUE)
str_detect(fnamesfake[1], "biz")

get_category <- function(title){
  if (str_detect(title, "biz")){
    return ("business")
  }
  else if (str_detect(title, "edu")){
    return ("education")
  }
  else if (str_detect(title, "entmt")){
    return ("entertainment")
  }
  else if (str_detect(title, "polit")){
    return ("politics")
  }
  else if (str_detect(title, "sports")){
    return ("sports")
  }
  else if (str_detect(title, "tech")){
    return ("technology")
  }
  else{
    return ("NA")
  }
}

get_category(fnamesfake[233])

create_news_data_frame <- function(fnames, x){ #fnames is the list.files, x is the value to add to new column (1 for fake, 0 for real)
  sample_df <- cbind(readLines(fnames[1])[1],readLines(fnames[1])[length(readLines(fnames[1]))], get_category(fnames[1]))
  for (i in (2:length(fnames))){
    all_lines <- readLines(fnames[i])
    the_row <- cbind(all_lines[1], all_lines[length(all_lines)], get_category(fnames[i]))
    sample_df <- rbind(sample_df, the_row)
  }
  sample_df <- as.data.frame(sample_df)
  names(sample_df)[names(sample_df) == "V1"] <- "title"
  names(sample_df)[names(sample_df) == "V2"] <- "text"
  names(sample_df)[names(sample_df) == "V3"] <- "category"
  sample_df <- mutate(sample_df, is_fake = x)
  return(sample_df)
}
fakenewsdf <- create_news_data_frame(fnamesfake, 1)
realnewsdf <- create_news_data_frame(fnamesreal, 0)
View(fakenewsdf)
View(realnewsdf)

complete_news_df_from_text <- rbind(fakenewsdf, realnewsdf)
View(complete_news_df_from_text)

#write.csv(complete_news_df_from_text,'fnd_news_compiled.csv')











