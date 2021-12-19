""" 
Importing Libraries

1) psutil and pandarallel for parallel processing a dataframe
2) tqdm for tracking processing
3) Sentence transformer for generating embeddings
4) Pandas and Numpy for dataframe processing
"""

import pickle
import re
import string
import spacy
import torch
from sentence_transformers import SentenceTransformer,  util
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm
from pandarallel import pandarallel
from pandarallel.utils import progress_bars
import psutil
workers = psutil.cpu_count()

progress_bars.is_notebook_lab = lambda: True

pandarallel.initialize(
    progress_bar=True, nb_workers=workers, use_memory_fs=False)

tqdm.pandas()


# ===============================================================================

"""

[Define script wide constants]

"""

WELFAKE_PATH = ""  # Path to the WELFake corpus
STOPWORDS_PATH = ""  # Path to SMART stopwords list
LEXICON_SAVE_PATH = ""  # Path to save the generated lexicon with scores
TOP_WORDS = 3000  # The number of top words to consider while creating the lexicon
# Should stopwords be removed before lexicon creation, if set to True will take more time
REMOVE_STOPWORDS = False

# ===============================================================================

"""
Creating the Lexicon

Defining cleaning/ preprocessing pipeline.
"""


def lower(text):
  """[Lowercase the string]

  Args:
      text ([String]): [Hello World]

  Returns:
      [String]: [hello world]
  """
  return text.lower()


def remove_urls(text):
  """[Remove URL from a string]

  Args:
      text ([String]): [Google url - google.com]

  Returns:
      [String]: [Google url - ]
  """
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  return url_pattern.sub(r'', text)


def remove_nonascii(sent):
  """[Remove non ascii characters from a string]

  Args:
      sent ([String])

  Returns:
      [String]
  """
  return "".join([i for i in sent if i.isascii()])


def remove_punctuations(text):
  """[Remove punctuations from a String]

  Args:
      text ([String]): [Hello World !!!]

  Returns:
      [String]: [Hello World]
  """
  res = re.sub(r'[^\w\s]', '', text)
  return res


def remove_num(text):
  """[Remove numbers from a String]

  Args:
      text ([String]): [I am 10 years old]

  Returns:
      [String]: [I am years old]
  """
  return "".join([c for c in text if not c.isdigit()])


def remove_mul_space(text):
  """[Removes multiple spaces from a String]

  Args:
      text ([String])

  Returns:
      [String]
  """
  return " ".join(text.split())


def clean(text):
  """[Implements all the above stated functions]

  Args:
      text ([String])

  Returns:
      [String]
  """

  text = lower(text)
  text = remove_urls(text)
  text = remove_nonascii(text)
  text = remove_punctuations(text)
  text = remove_num(text)
  text = remove_mul_space(text)

  return text

# ===============================================================================


def createStopWords(path):
  """[Import the SMART Stop words list]

  Args:
      path ([String]): [The path to the local stop words file]
  """
  stopwords = []
  with open(path, "r") as f:
    for word in f:
      stopwords.append(word.strip().replace("'", ""))


stop = createStopWords(STOPWORDS_PATH)

# ===============================================================================


def getDataFrames(path, true=0, fake=1):
  """[Get true and fake split of dataframe]

  Args:
      path ([String]): [Path to the WELFake Lexicon]
      true (int, optional): [Label of the True column]. Defaults to 0.
      fake (int, optional): [Label of the Fake column]. Defaults to 1.

  Returns:
      [DataFrames]: [df_true and df_fake dataframes]
  """
  df = pd.read_csv(path)

  df_fake = df[df['label'] == fake].copy(deep=True)
  df_true = df[df['label'] == true].copy(deep=True)

  df_true['total_text'] = df_true['title'].fillna(
      '') + " " + df_true['text'].fillna('')
  df_fake['total_text'] = df_fake['title'].fillna(
      '') + " " + df_fake['text'].fillna('')

  df_true['total_text'] = df_true['total_text'].parallel_apply(clean)
  df_fake['total_text'] = df_fake['total_text'].parallel_apply(clean)

  df_true = df_true.drop_duplicates(subset=["total_text"])
  df_fake = df_fake.drop_duplicates(subset=["total_text"])

  return df_true, df_fake


df_true, df_fake = getDataFrames(WELFAKE_PATH)

# ===============================================================================

"""

[Lemmatizing the corpus before lexicon creation]

"""

nlp_lemmatize = spacy.load(
    "en", disable=['parser', 'ner', 'tagger', 'textcat'])

df_true["total_text_lemmatized"] = df_true["total_text"].parallel_apply(
    lambda row: " ".join([w.lemma_ for w in nlp_lemmatize(row)]))
df_fake["total_text_lemmatized"] = df_fake["total_text"].parallel_apply(
    lambda row: " ".join([w.lemma_ for w in nlp_lemmatize(row)]))

# ===============================================================================


def createLexicon(df_true, df_fake, top_words, remove_stop):
  """[summary]

  Args:
      df_true ([type]): [description]
      df_fake ([type]): [description]
      top_words ([type]): [description]
      remove_stop (bool, optional): [description]. Defaults to False.

  Returns:
      [type]: [description]
  """

  true_data = df_true['total_text_lemmatized'].tolist()
  fake_data = df_fake['total_text_lemmatized'].tolist()

  fake_words, true_words = [], []

  for item in fake_data:
    fake_words.extend(item.strip().split())
  for item in true_data:
    true_words.extend(item.strip().split())

  if(remove_stop):
    fake_words = [word for word in fake_words if (
        len(word) >= 3) and (word not in stop)]
    true_words = [word for word in true_words if (
        len(word) >= 3) and (word not in stop)]
  else:
    fake_words = [word for word in fake_words if (len(word) >= 3)]
    true_words = [word for word in true_words if (len(word) >= 3)]

  counts_common, counts_true, counts_fake = {}, {}, {}

  counts_fake = dict(Counter(fake_words))
  counts_true = dict(Counter(true_words))

  counts_true = dict(sorted(counts_true.items(),
                            key=lambda item: item[1], reverse=True)[:top_words])
  counts_fake = dict(sorted(counts_fake.items(),
                            key=lambda item: item[1], reverse=True)[:top_words])

  for key in counts_true.keys():
    if key in counts_fake.keys():
      counts_common[key] = counts_true[key] + counts_fake[key]

  nlp = spacy.load("en", disable=['parser', 'ner', 'lemmatizer', 'textcat'])

  nlp.max_length = 90000000

  fake_words_nouns = []
  true_words_nouns = []

  doc1 = nlp(' '.join(list(counts_fake.keys())))
  for token in doc1:
    if(token.tag_ in ['NNP', 'NNPS']):
      fake_words_nouns.append(str(token))

  doc2 = nlp(' '.join(list(counts_true.keys())))
  for token in doc2:
    if(token.tag_ in ['NNP', 'NNPS']):
      true_words_nouns.append(str(token))

  issue = []

  for word in fake_words_nouns:
    try:
      del counts_fake[word]
      try:
        del counts_common[word]
      except KeyError:
        continue

    except KeyError:
      issue.append(word)

  for word in true_words_nouns:
    try:
      del counts_true[word]
      try:
        del counts_common[word]
      except KeyError:
        continue

    except KeyError:
      issue.append(word)

  print(issue)

  if '-PRON-' in counts_fake.keys():
    del counts_fake['-PRON-']

  if '-PRON-' in counts_true.keys():
    del counts_true['-PRON-']

  if '-PRON-' in counts_common.keys():
    del counts_common['-PRON-']

  doc_true_words = {}

  for key in counts_true.keys():
    occ = 0
    for data in true_data:
      if key in data:
        occ += 1
    doc_true_words[key] = occ

  doc_fake_words = {}

  for key in counts_fake.keys():
    occ = 0
    for data in fake_data:
      if key in data:
        occ += 1
    doc_fake_words[key] = occ

  doc_common_words = {}

  for key in counts_common.keys():
    doc_common_words[key] = doc_fake_words[key] + doc_true_words[key]

  words = list(counts_true.keys())+list(counts_fake.keys()) + \
      list(counts_common.keys())
  words = list(set(words))

  fake_score, true_score, common_score = [], [], []
  doc_fake_score, doc_true_score, doc_common_score = [], [], []

  for word in words:

    if word in doc_true_words.keys():
      doc_true_score.append(doc_true_words[word])
    else:
      doc_true_score.append(0)

    if word in doc_fake_words.keys():
      doc_fake_score.append(doc_fake_words[word])
    else:
      doc_fake_score.append(0)

    if word in doc_common_words.keys():
      doc_common_score.append(doc_common_words[word])
    else:
      doc_common_score.append(0)

    if word in counts_true.keys():
      true_score.append(counts_true[word])
    else:
      true_score.append(0)

    if word in counts_common.keys():
      common_score.append(counts_common[word])
    else:
      common_score.append(0)

    if word in counts_fake.keys():
      fake_score.append(counts_fake[word])
    else:
      fake_score.append(0)

  words_and_scores = pd.DataFrame()
  words_and_scores['word'] = words
  words_and_scores['common_score'] = common_score
  words_and_scores['true_score'] = true_score
  words_and_scores['fake_score'] = fake_score
  words_and_scores['doc_common_score'] = doc_common_score
  words_and_scores['doc_true_score'] = doc_true_score
  words_and_scores['doc_fake_score'] = doc_fake_score

  return words_and_scores


words_and_scores = createLexicon(df_true, df_fake, TOP_WORDS, REMOVE_STOPWORDS)
words_and_scores.to_csv(LEXICON_SAVE_PATH, index=False)
