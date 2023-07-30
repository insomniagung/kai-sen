import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import re
from unidecode import unidecode

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
from nlp_id import StopWord, Tokenizer    
   
# 2. Data Cleaning
def casefolding(text):
    text = text.lower()
    text = unidecode(text)
    return text

def cleansing(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Data Normalize
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(text):
    stemmed_text = stemmer.stem(text)
    return stemmed_text

kbba_dictionary = pd.read_csv(
            'https://raw.githubusercontent.com/insomniagung/kamus_kbba/main/kbba.txt', 
            delimiter='\t', names=['slang', 'formal'], header=None, encoding='utf-8')

slang_dict = dict(zip(kbba_dictionary['slang'], kbba_dictionary['formal']))
def convert_slangword(text):
    words = text.split()
    normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]
    normalized_text = ' '.join(normalized_words)
    return normalized_text

# 4. Words Removal
def remove_stopword(text):
    stopword = StopWord()
    text = stopword.remove_stopword(text)
    return text

def remove_unwanted_words(text):
    unwanted_words = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 
                      'sep', 'oct', 'nov', 'dec', 'januari', 'februari', 'maret', 
                      'april', 'mei', 'juni', 'juli', 'agustus', 'september', 
                      'oktober', 'november', 'desember', 'gin'}
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in unwanted_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def remove_short_words(text):
    return ' '.join([word for word in text.split() if len(word) >= 3])

# 5. Tokenizing
tokenizer = Tokenizer()
def tokenizing(text):
    return tokenizer.tokenize(text)

# Dictionary kata positif yang digunakan :
df_positive = pd.read_csv(
    'https://raw.githubusercontent.com/SadamMahendra/ID-NegPos/main/positive.txt', sep='\t')
list_positive = list(df_positive.iloc[::, 0])

# Dictionary kata negatif yang digunakan :
df_negative = pd.read_csv(
    'https://raw.githubusercontent.com/SadamMahendra/ID-NegPos/main/negative.txt', sep='\t')
list_negative = list(df_negative.iloc[::, 0])

def sentiment_analysis_dictionary_id(text):
    for word in text:
        if word in list_positive:
            score += 1
        if word in list_negative:
            score -= 1

    polarity = ''
    if (score > 0):
        polarity = 'positive'
    elif (score < 0):
        polarity = 'negative'
    else:
        polarity = 'neutral'

    return score, polarity
