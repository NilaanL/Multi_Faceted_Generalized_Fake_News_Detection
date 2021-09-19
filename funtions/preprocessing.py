from langdetect import detect
import numpy as np
import pandas as pd
import re

def detect_lang(text):
  try:
      return detect(text)
  except:
      return 'error'
# -------------------------------------------------------------------------------

#remove non-ascii
def remove_nonascii(sent):
  return " ".join(("".join([i for i in sent if i.isascii()])).split())
# -------------------------------------------------------------------------------

# remove urls
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
# -------------------------------------------------------------------------------

def count_digits(text):
    '''
    (str)->int
    Taking digits count 
    count afte url removal
    '''
  return sum(c.isdigit() for c in text)
# -------------------------------------------------------------------------------

def count_numbers(text):
    '''
    (str)->int
    Taking numbers count 
    count afte url removal
    '''
  tokens = text.split()
  return sum(c.isnumeric() for c in tokens)
# -------------------------------------------------------------------------------

puncList = ['!', "," ,"\'" ,";" ,"\"", ".", "-" ,"?",':','_']
stop = []
with open("./resources/SMART_STOP_WORDS.txt", "r") as f:
  for word in f:
      # Here we remove the apostrophe as well
      stop.append(word.strip().replace("'",""))

def count_stopwords(text):
  stop_words = 0
  updated_text = ''.join(i for i in text if i not in puncList)
  tokens = updated_text.split()
  for token in tokens:
    if (token in stop):
      stop_words +=1
  return stop_words
# -------------------------------------------------------------------------------
def remove_stopwords(text):    
    updated_text = ''.join(i for i in text if i not in puncList)
    tokens = updated_text.split()
    return " ".join([word for word in tokens if word not in stop])
# -------------------------------------------------------------------------------

def strip_all_tags(text):
    '''
    uses this to remove any mentions and hash tags
    '''
    entity_prefixes = ['@','#']
    words = []
    for word in text.split():
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
# -------------------------------------------------------------------------------

def rem_mul_space(text):
    '''
    remove multiple spaces
    '''
  return " ".join(text.split())
# -------------------------------------------------------------------------------

def remove_emoji(string):
    '''
    Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    remove emojis
    '''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
# -------------------------------------------------------------------------------
