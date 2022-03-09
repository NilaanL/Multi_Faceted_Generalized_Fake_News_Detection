import functions.emotion_feature
import sementic_feature
import functions.lexicon_creation
import functions.postprocessing
import functions.preprocessing
import functions
import config
from tqdm.auto import tqdm
import functions.helper


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tqdm.pandas()

DATASET_PATH=config.dataset_path
TEXTCOLUMN = config.TEXT
IDCOLUMN = config.ID
LABELCOLUMN = config.LABEL

FEATURE_RESULT_PATH=config.feature_result_path
 

def get_sementic_features(dataframe, TEXTCOLUMN="text" ,LABELCOLUMN="label" ):
    df=dataframe.copy(deep=True)

    TEXT  = TEXTCOLUMN
    LABEL = LABELCOLUMN
    df["text"] = df[TEXTCOLUMN]
    df["label"] = df[LABEL]

    unnamed=df.columns[df.columns.str.contains('unnamed',case = False)]
    df.drop(columns=unnamed,inplace=True)

    df = df.dropna(subset=['text'], how='all')
    df = df.reset_index(drop=True)
    df['text'] = df['text'].replace(np.nan, '', regex=True)
    df = df.dropna(subset=['text'], how='all') 
    df.info()

    df["url_count"] = df["text"].progress_apply(url_count)
    df["text"] = df["text"].progress_apply(remove_urls)
    df["text"] = df["text"].progress_apply(remove_nonascii) 

    df = df.dropna(subset=['text'], how='all')

    df["qn_symbol"]=df["text"].apply(get_no_of_qn_marks)
    
    df["num_chars"]=df["text"].progress_apply(num_chars)
    df["num_words"]=df["text"].progress_apply(num_words)
    df["num_sentences"]=df["text"].progress_apply(num_sentences)
    df["words_per_sentence"]=df["text"].progress_apply(words_per_sentence)
    df["characters_per_word"]=df["text"].progress_apply(characters_per_word)
    df["punctuations_per_sentence"]=df["text"].progress_apply(punctuations_per_sentence)

    df["positive"]=df["text"].progress_apply(positive)
    df["negative"]=df["text"].progress_apply(negative)

    df["num_exclamation"]=df["text"].progress_apply(num_exclamation)
    df["get_sentiment_polarity"]=df["text"].progress_apply(get_sentiment_polarity)
    df["lexical_diversity"]=df["text"].progress_apply(lexical_diversity)

    df["content_word_diversity_and_redundancy"]=df["text"].progress_apply(content_word_diversity_and_redundancy)
    df["content_word_diversity"]=df["content_word_diversity_and_redundancy"].progress_apply(lambda x: x["content_word_diversity"])
    df["redundancy"]=df["content_word_diversity_and_redundancy"].progress_apply(lambda x: x["redundancy"])
    
    df["nvaa"]=df["text"].progress_apply(nvaa)
    df["noun"]=df["nvaa"].progress_apply(lambda x: x["NOUN"])
    df["verb"]=df["nvaa"].progress_apply(lambda x: x["VERB"])
    df["adj"]=df["nvaa"].progress_apply(lambda x: x["ADJ"])
    df["adv"]=df["nvaa"].progress_apply(lambda x: x["ADV"])

    df["lang"]=df["text"].progress_apply(detect_lang)

    df["qn_symbol_per_sentence"]=df["qn_symbol"]/df["num_sentences"]
    df["num_exclamation_per_sentence"]=df["num_exclamation"]/df["num_sentences"]
    df["url_count_per_sentence"]=df["url_count"]/df["num_sentences"]

    return df 

for key,value in DATASET_PATH.items():
    df=pd.read_csv(value)
    df_sementic = get_sementic_features(df,TEXTCOLUMN[key],LABELCOLUMN[key])
    df.to_csv(FEATURE_RESULT_PATH[key]["semantic"],index=False)


