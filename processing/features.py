import nltk
from textblob import TextBlob
from collections import Counter

#This function calculates the average length of text and puts them in a vector per row in the dataframe
def average_length(text):
    average_length_list = []
    for txt in text:
        sentences = sent_tokenize(txt) #splitting into sentences
        num_words = [len(sentence.split(' ')) for sentence in sentences] #numwords per sentence
        average = sum(num for num in num_words)/(len(sentences)) #average number of words / sentence for 
        average_length_list.append(average) #len words per text 
    return average_length_list

#this function counts the number of exclamations per dataset and creates a vector of values 

def exclamations(text):
    exclamations_list = []
    for txt in text:
        count = 0
        for i in range(len(txt)):
            if txt[i] in ('!',"?"):  
                count = count + 1;  
        exclamations_list.append(count)
    return exclamations_list

#this function calculates the frequency of superlatives in the text 
#superlatives are comparison adverbs/adjectives. For example, "best" vs "better" where "best" is considered a superlative

def textblob_adj(text):
    blobed = TextBlob(text)
    counts = Counter(tag for word,tag in blobed.tags)
    super_list = []
    super_tag_list = ['JJS','RBS']
    for (a, b) in blobed.tags:
        if b in super_tag_list:
           super_list.append(a)
        else:
            pass
    return counts['JJS'] + counts['RBS']

#list_superlatives creates a vector of counts of superlatives per text

def list_superlatives(text):
    superlative_list = []
    for txt in text:
        superlative_list.append(textblob_adj(txt))
    return superlative_list

