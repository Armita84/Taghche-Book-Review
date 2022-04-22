# -*- coding: utf-8 -*-
"""Preprocess_Taghcheh"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("taghche.csv")
df

# dataset statistics summary
df.describe()

"""## Check For Missing/Null Values"""

# number of missing values in each column
df.isnull().sum()

# drop all null/ missing values for classification
df1 = df.dropna()

df1.isnull().sum()

"""## Data Cleaning"""

# install Hazm Persian/Farsi Preprocessing text package
!pip install hazm

from __future__ import unicode_literals
from hazm import *
import re

#normalize the text
normalizer = Normalizer()

def normal(text):
    text=str(text)
    text = normalizer.character_refinement(text)
    text = normalizer.punctuation_spacing(text)
    text = normalizer.affix_spacing(text)
    text = normalizer.normalize(text)
    return text

#find stopwords in stopwords file
stopwords = []
file = open('stopwords-fa-new.txt').read()
[stopwords.append(x) for x in file.split('\n')]
stopwords = set(stopwords)
len(stopwords)

def remove_stopwords(text):
    text=str(text)
    filtered_tokens = [token for token in text.split() if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_emoji(text): 
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u200c"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text)

def remove_halfspace(text): 
    emoji_pattern = re.compile("["                
                u"\u200c"              
    "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text) 

def remove_link(text): 
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))
    # return re.sub(r"\bhttps?://\S+", "", str(text))

def remove_picUrl(text):
    return re.sub(r'pic.twitter.com/[\w]*',"", str(text))

def remove_rt(text):
    z = lambda text: re.compile('\#').sub('', re.compile('RT @').sub('@', str(text), count=1).strip())
    return z(text)

def remove_hashtag(text):
    return re.sub(r"#[^\s]+", '', str(text))

def remove_mention(text):
    return re.sub(r"@[^\s]+", '', str(text))

def remove_email(text): 
    return re.sub(r'\S+@\S+', '', str(text))

def remove_numbers(text): 
    return re.sub(r'^\d+\s|\s\d+\s|\s\d+$', ' ', str(text))

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', str(text))

def remove_quote(text): 
    return  str(text).replace("'","")

def remove_chars(text): 
    # return re.sub(r'\.(?!\d)', '', str(text))
    return  re.sub(r'[$+&+;+]|[><!+،:’,\(\).+]|[-+]|[…]|[\[\]»«//]|[\\]|[#+]|[_+]|[—+]|[*+]|[؟+]|[?+]|[""]', ' ', str(text))

def remove_englishword(text): 
    return re.sub(r'[A-Za-z]+', '', str(text))

def remove_extraspaces(text):
    return re.sub(r' +', ' ', text)

def remove_extranewlines(text):
    return re.sub(r'\n\n+', '\n\n', text)

#lemmatize the text
lemmatizer = Lemmatizer()

def lemma(text):
    text=str(text)
    return lemmatizer.lemmatize(text)

def preprocess2(text):
    text = remove_link(text)
    text = remove_picUrl(text)
    text = remove_englishword(text)
    text = normal(text)
    text = remove_stopwords(text)
    text = remove_emoji(text)
    text = remove_rt(text)
    text = remove_mention(text)
    text = remove_emoji(text)
    text = remove_hashtag(text)   
    text = remove_email(text) 
    text = remove_html(text) 
    text = remove_chars(text)
    text = remove_numbers(text)
    text = remove_quote(text)
    text = lemma(text)
    text = remove_extraspaces(text)
    text = remove_extranewlines(text)
    text = remove_halfspace(text) 
    text = remove_stopwords(text)
    return text

df1_cleaned = list(map(preprocess2, df1["comment"]))

df1_cleaned[0:30]

df1["cleaned_comment"] = df1_cleaned

df1.head(30)

"""## Data Analysis & Data Visualization

"""

# Add a new feature for lenght of comments
df1['comment_length'] = df1['cleaned_comment'].astype(str).apply(len)

# Add a new feature for word count in comments
df1['word_count'] = df1['cleaned_comment'].apply(lambda x: len(str(x).split()))

df1.head()

df1.describe()

# Comments count based on the rating distribution
pd.options.plotting.backend = "plotly"

df1['rate'].plot(
    kind='hist',
    bins=50,
    x="rate",
    title='Rating Distribution',
    color_discrete_sequence = ['purple']
    )

# sort dataframe based on rate score from top to bottom
df1.sort_values(by=['rate'], ascending=False)

# comments count based on comment length distribution
print("comments count based on comment length distribution")
df1.groupby('comment_length')['comment'].nunique().plot(kind='line')

# comments count based on the comment Rate and Likes number distribution
print("comments count based on the comment Rate and Likes number distribution")
import matplotlib.pyplot as plt
df1.groupby(['rate','like']).size().unstack().plot(kind='bar')

import matplotlib.ticker as mtick

# comments count based on the comment Rate and Likes number distribution In PERCENTAGE
print("comments count based on the comment Rate and Likes number distribution In PERCENTAGE")
df1.groupby(['rate','like']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar', )

# Comments count based on the Date distribution
df1['date'].plot(
    kind='hist',
    bins=50,
    x="date",
    title='Date Distribution',
    color_discrete_sequence = ['lightskyblue']
    )

#Count Comments for each Book Name
df1['bookname'].plot(
    kind='hist',
    bins=150,
    x="bookname",
    title='Count of comments for each Book',
    color_discrete_sequence = ['lightskyblue']
    )

# Find top unigrams in comments 
from sklearn.feature_extraction.text import CountVectorizer

def n_grams(documents, number=None):
    vectorized = CountVectorizer().fit(documents)
    bag_of_words = vectorized.transform(documents)
    sum = bag_of_words.sum(axis=0) 
    freq = [(word, sum[0, idx]) for word, idx in vectorized.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:number]

common_words = n_grams(df1['cleaned_comment'], 30)
# for word, w_freq in common_words:
#     print(word, w_freq)
df2 = pd.DataFrame(common_words, columns = ['cleaned_comment' , 'count'])
df2.groupby('cleaned_comment').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 words in comments')

# Find top bigrams in comments 
from sklearn.feature_extraction.text import CountVectorizer

def n_grams(documents, number=None):
    vectorized = CountVectorizer(ngram_range=(2,2)).fit(documents)
    bag_of_words = vectorized.transform(documents)
    sum = bag_of_words.sum(axis=0) 
    freq = [(word, sum[0, idx]) for word, idx in vectorized.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:number]

common_words = n_grams(df1['cleaned_comment'], 30)
# for word, w_freq in common_words:
#     print(word, w_freq)
df2 = pd.DataFrame(common_words, columns = ['cleaned_comment' , 'count'])
df2.groupby('cleaned_comment').sum()['count'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 Bigrams in comments')

# Count number of commnets submitted for each book
print("Count number of commnets submitted for each book:")
df1.groupby(['bookname'])['comment'].nunique().sort_values(ascending=False).reset_index(name='count').head(20)

# Top 10 Book names based on most comments
print("Top 10 Book names based on most comments:")
df1.groupby(['bookname'])['comment'].nunique().sort_values(ascending=False).reset_index(name='count').head(10)

# Top 10 Book IDs based on most comments
print("Top 10 Book IDs based on most comments:")
df1.groupby(['bookID'])['comment'].nunique().sort_values(ascending=False).reset_index(name='count').head(10)

# 10 Book names based on less comments
print("10 Book names based on less comments:")
df1.groupby(['bookname'])['comment'].nunique().sort_values(ascending=True).reset_index(name='count').head(10)

# 10 Book IDs based on less comments
print("10 Book IDs based on less comments:")
df1.groupby(['bookID'])['comment'].nunique().sort_values(ascending=True).reset_index(name='count').head(10)

#Like distribution
df1['like'].plot(
    kind='hist',
    x= "like",
    title='Count of comments based on Likes number'
    )

# sort dataframe based on rate score from top to bottom
df1.sort_values(by=['like'], ascending=False)

df1.describe()

# Top 10 Book names based on most likes
print("Top 10 Book names based on most Likes:")
df1.groupby(['bookname'])['like'].nunique().sort_values(ascending=False).reset_index(name='count').head(10)

# Top 10 Book IDs based on most likes
print("Top 10 Book IDs based on most Likes:")
df1.groupby(['bookID'])['like'].nunique().sort_values(ascending=False).reset_index(name='count').head(10)

# 10 Book names based on less likes
print("10 Book names based on less Likes:")
df1.groupby(['bookname'])['like'].nunique().sort_values(ascending=True).reset_index(name='count').head(10)

# 10 Book IDs based on less likes
print("10 Book IDs based on less Likes:")
df1.groupby(['bookID'])['like'].nunique().sort_values(ascending=True).reset_index(name='count').head(10)

"""# Word Cloud"""

!pip install persian_wordcloud

from os import path
from persian_wordcloud.wordcloud import PersianWordCloud, add_stop_words
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Get Stopwords
stopwords = []
file = open('stopwords-fa-new.txt').read()
[stopwords.append(x) for x in file.split('\n')]
stopwords = set(stopwords)

#Get Most Frequent Words in a tuple format ('count','word')
from collections import Counter
freq_words = Counter(" ".join(df1["cleaned_comment"]).split()).most_common(100)

# Get a word from the tuple
n = 0 # N. . .
terms = [x[n] for x in freq_words]

# Write frequent terms in a text file
with open("frequent_words.txt", "w") as output:
    output.write(str(terms))

# Read frequent terms from the file
g_terms = open('frequent_words.txt', encoding='utf-8').read()

# Create the word Cloud
wordcloud = PersianWordCloud(
    only_persian=True,
    max_words=100,
    stopwords=stopwords,
    margin=0,
    width=1000,
    height=1000,
    min_font_size=1,
    max_font_size=500,
    background_color="white",
).generate(g_terms)

image = wordcloud.to_image()
image.show()

# Save the Word Cloud as an image
image.save('frequent_words.png')

# Read the Word Cloud image
img=mpimg.imread('frequent_words.png')
plt.figure(figsize = (10,10))
imgplot = plt.imshow(img)
plt.show()