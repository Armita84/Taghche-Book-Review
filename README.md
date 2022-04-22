# Taghche-Book-Review
Data Analysis/Visualization, and BERT Binary-classification for Taghche Dataset (Persian Book comments and reviews)

# List of Files/Folders in the Repo:

##    - stopwords-fa-new.txt
        This file contains the common list of Farsi/Persian Stopwords.
    
##    - taghche.csv
        Taghche dataset (an online platform and app for reading Persian Ebooks) in CSV format with the below features:
            *date (the date of book comments)
            *comment (the review of the books by readers)
            *bookname (name of the book)
            *rate (rate the book from 0 to 5 - 0 means no rate)
            *bookID (book Identification Number)
            *like (Number of Likes by reader for each comment)
        
        (You can find and download the original dataset from https://www.kaggle.com/saeedtqp/taaghche)
    
##    - preprocess_taghcheh.py
        In this file, Taghche data is cleaned and preprocessed using HAZM Preprocessing Package (For Farsi language).
        There are some data visualizations using matplotlib plotly visualization tool.
        Some analysis regarding the comments length, unigram and bi-grams features in the comments,the comments rating and the number of likes for each comment has done in this file.
        Also, the most frequent words in comments is shown via Persina Word Cloud. 
    
##    - bert_finetune_taghcheh.py
        Using 'bert-base-multilingual-uncased' model with pytorch, the data is finetuned and classified. I did a binary classification as below:
        Consider the comments' rate numbers 4 and 5 to the target/class 1, and the rate numbers less or equal to 3 as the class of 0.
        The number of comments with the rate 5 were much more than the comments with other ratings, so I sampled a less comments of rate 5 to have a balanced data before converting it to a        bi-class.
