# Functions for feature extraction, based on **Table 4** from **Zhou and Zafarani, 2020**. The semantic level features are divided in to the following broad categories,


# ---


# 1.   Quantity
# 2.   Complexity
# 3.   Uncertainity
# 4.   Subjectivity
# 5.   Non-immediancy
# 6.   Sentiment
# 7.   Diversity
# 8.   Informality
# 9.   Specificity
# 10.  Readability


# ---

# # Following are the required libraries for this notebook (install via requiremnts.txt)

# ! pip install textstat -q
# ! pip install lexical-diversity -q
# ! pip install spacy -q
# ! python -m spacy download en_core_web_sm -q
# ! pip install vaderSentiment -q
# ! pip install transformers -q

# ===============================================================================
# importing the necessary libraries

import textstat
import re

import nltk
nltk.download("punkt")

import string

from lexical_diversity import lex_div as ld

import spacy
nlp_tagger = spacy.load('en_core_web_sm')
nlp_tagger.disable_pipes('parser', 'ner')

from spacy.lang.en import English
nlp_stop = English()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
from spacy.lang.en.stop_words import STOP_WORDS

# ===============================================================================
'''Funtion for sementic feature extraction ''''

"""
1) Quantity

{Quantity includes the following features,


*   Number of characters
*   Number of words
*   Number of Noun Phrases
*   Number of sentences
*   Number of paragraphs

Out of the above features, we are only interested in **characters, words and sentences**.

"""

def num_chars(text):
    '''(str) -> number    #**TypeContract**
    returns the number of characters in a text.  #**Description**
    '''
  return len(text)
# -------------------------------------------------------------------------------

#get url count
def url_count(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls=re.findall(url_pattern,text)
    return len(urls)
# -------------------------------------------------------------------------------

# get qn mark count
def get_no_of_qn_marks(text,percentage=False):
  return text.count("?")/len(text) if percentage else text.count("?")
# -------------------------------------------------------------------------------

# This function calculates the number of words in the text, **excluding the punctuations.
def num_words(text):
  return textstat.lexicon_count(text, removepunct=True)
# -------------------------------------------------------------------------------

# This function calculates the number of sentences in the text.
def num_sentences(text):
  return textstat.sentence_count(text)
# -------------------------------------------------------------------------------

# ===============================================================================
"""
2) Complexity

Complexity includes the following features,

*   Average number of characters per word
*   Average number of words per sentence
*   Average number of clauses per sentence
*   Average number of punctuations per sentence
"""

# The following function calculates the average number of words per sentence, 
# i.e. (number of words/number of sentences)
def words_per_sentence(text):
  return float(num_words(text))/num_sentences(text)
# -------------------------------------------------------------------------------

# The function calculates the number of characters per word.
def characters_per_word(text):
  tokens = nltk.word_tokenize(text)
  nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
  filtered = [w for w in tokens if nonPunct.match(w)]
  return float(sum(map(len, filtered))) / len(filtered) if len(filtered)>0 else 0
# -------------------------------------------------------------------------------

# This function calculates the average number of punctuations per sentence.
def punctuations_per_sentence(text):
  punc_count = sum([1 if char in string.punctuation else 0 for char in text])
  return punc_count / float(num_sentences(text))
# -------------------------------------------------------------------------------

# ===============================================================================

"""
3) Sentiment

Sentiment includes the following features,

*  Percentage of Positive words
*  Percentage of Negative words
*  Number of Exclamation marks
*  Content Sentiment Polarity
*  Percentage of Anxiety/angry/sadness words
"""

"""
3.1) Percentage of Positive words

This function calculates the amount of postive words in the sentence as a percentage.

The function uses a corpus comparison method, the corpus is from, **Minqing Hu and Bing Liu. 
"Mining and Summarizing Customer Reviews.", Proceedings of the ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA.
"""

# First we read the word list from drive
with open("./resources/word_lists/positive_words.txt") as f:
    positive_words = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
positive_words = [x.strip() for x in positive_words]
# -------------------------------------------------------------------------------

def positive(text):
  tokens = nltk.word_tokenize(text)
  nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
  filtered = [w for w in tokens if nonPunct.match(w)]

  count = 0
  for word in filtered:
    if word in positive_words:
      count+=1

  return (float(count)/len(filtered))*100  if len(filtered)>0 else 0
# -------------------------------------------------------------------------------

"""
3.2) Percentage of Negative words

This function calculates the amount of negative words in the sentence as a percentage.
"""
# First we read the word list from drive 
with open("./resources/word_lists/negative_words.txt" ,encoding="utf-8" ,  errors="ignore") as f:
    negative_words = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
negative_words = [x.strip() for x in negative_words]
# -------------------------------------------------------------------------------

def negative(text):
  tokens = nltk.word_tokenize(text)
  nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
  filtered = [w for w in tokens if nonPunct.match(w)]

  count = 0
  for word in filtered:
    if word in negative_words:
      count+=1

  return (float(count)/len(filtered))*100  if len(filtered)>0 else 0
# -------------------------------------------------------------------------------

# This function calculates the number of exclamation marks in the text
def num_exclamation(text):
  tokens = nltk.word_tokenize(text)
  return len([w for w in tokens if w == "!"])
# -------------------------------------------------------------------------------

"""
3.4) Content Sentiment Polarity

This is calculated using [VaderSentiment](https://github.com/cjhutto/vaderSentiment).

The compound score is computed by summing the valence scores of each word in the lexicon, 
adjusted according to the rules, and then normalized to be between -1 (most extreme negative) 
and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional 
measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.
"""

def get_sentiment_polarity(text):
  vader_scores = sentiment_analyzer.polarity_scores(text)
  return vader_scores['compound'] # Outputs something like this {'pos': 0.094, 'compound': -0.7042, 'neu': 0.579, 'neg': 0.327}
# -------------------------------------------------------------------------------

# ===============================================================================

"""
4) Diversity**

Diversity includes the following features,
*   Lexical diversity
*   Content word diversity
*   Redundancy
*   Unique Nouns/ Verbs/ Adjectives/ Adverbs
"""

"""
Lexical Diversity 
This function calculates the TTR - TTR is the ratio obtained by dividing the types 
(the total number of different words) occurring in a text or utterance by its tokens 
(the total number of words).This function is written using the pypi lexical-diversity.
"""

'''
Lemmatize is by default set to False, but if we want to lemmatize we could set 
that to True.

The lemmatizer which is not part of speech specific ('run' as a noun and 'run' 
as a verb are treated as the same word). However, it is likely better to use a 
part of speech sensitive lemmatizer (e.g., using spaCy).
'''

def lexical_diversity(text, lemmatize = False):
  tokens = ld.flemmatize(text) if lemmatize else ld.tokenize(text)
  return ld.ttr(tokens) * 100
# -------------------------------------------------------------------------------

def content_word_diversity_and_redundancy(text):

  text = re.sub(r'[^\w\s]', '', text)
  doc = nlp_stop(text)

  # Create list of word tokens
  token_list = []
  for token in doc:
      token_list.append(token.text)

  # Create list of word tokens after removing stopwords
  content_words, function_words =[], []
  for word in token_list:
      lexeme = nlp_stop.vocab[word]
      if lexeme.is_stop == False:
          content_words.append(word) 
      else:
          function_words.append(word)
  
  output = {
      'content_word_diversity': (float(len(list(set(content_words)))) / num_words(text)) * 100  if num_words(text)>0 else 0 ,
      'redundancy': (float(len(list(set(function_words)))) / num_words(text)) * 100 if num_words(text)>0 else 0 ,
  }
  
  return output
# -------------------------------------------------------------------------------


# This function calculates the percentage of unique nouns, verbs, adjectives and adverbs and returns the results as a list.

def nvaa(text):
  # Regular expression to take out the punctuations
  text = re.sub(r'[^\w\s]', '', text)

  doc = nlp_tagger(text)

  pos_tags = {
      'NOUN': [],
      'VERB': [],
      'ADJ': [],
      'ADV': [],
  }

  keys = pos_tags.keys()

  # Iterate over the tokens
  for token in doc:
      pos = token.pos_
      if pos in keys:
        pos_tags[pos].append(token.text)
  # print(pos_tags)

  output = {
      'NOUN': 0,
      'VERB': 0,
      'ADJ': 0,
      'ADV': 0
  }

  for key in output.keys():
    output[key] = (len(list(set(pos_tags[key]))) / float(num_words(text))) * 100 if num_words(text)>0 else 0

  return output
# -------------------------------------------------------------------------------
