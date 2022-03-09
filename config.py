sementic_features= [
                    'url_count',
                    'qn_symbol',
                    'num_chars',
                    'num_words',
                    'num_sentences',
                    'words_per_sentence',
                    'characters_per_word',
                    'punctuations_per_sentence',
                    'num_exclamation',
                    'get_sentiment_polarity',
                    'lexical_diversity',
                    'content_word_diversity',
                    'redundancy',
                    'noun',
                    'verb',
                    'adj',
                    'adv',
                    "qn_symbol_per_sentence",
                    "num_exclamation_per_sentence",
                    "url_count_per_sentence"
                    ]
LangMod_Features=   [
                     'fake_score', 
                    'true_score', 
                    'common_score'
                    ]

Sentiment_features= [
                    #  'highest_eight_label', 
                    'anger', 
                    'anticipation',
                    'disgust', 
                    'fear',
                    'joy', 
                    'sadness', 
                    'surprise', 
                    'trust'
                    ]
All_features=sementic_features+LangMod_Features+Sentiment_features

dataset_path={
    "codalab": "/Datasets/CodaLab Covid/Constraint_English_All.csv",
    "fakenewsnet":"/Datasets/FakeNewsNet/FakeNewsNet_All.csv",
    "isot":"/Datasets/ISOT/ISOT.csv",
    # "kaggle":"/Datasets/Kaggle/Kaggle.csv",
    "kagglerealfake":"/Datasets/Kaggle_real_fake/fake_or_real_news.csv",    
    "liar": "/Datasets/LIAR/Liar_all.csv",
    # "politifact":"/Datasets/Politifact/Politifact.tsv",
    # "welfake":"/Datasets/WELFAKE/WELFake.csv"
}

feature_result_path={
    "codalab":
        {
        "lexicon" :"/features/CodaLab Covid/CodaLab_lexicon.csv",
        "semantic":"/features/CodaLab Covid/CodaLab_sementic.csv",
        "emotion" :"/features/CodaLab Covid/CodaLab_emotions.csv",
        "embedding" : "/features/CodaLab Covid/CodaLab_embedding.csv"
        },
    "fakenewsnet" :
        {
        "lexicon"  :"/features/FakeNewsNet/FakeNewsNet_lexicon.csv",
        "semantic" :"/features/FakeNewsNet/FakeNewsNet_sementic.csv",
        "emotion"  : "/features/FakeNewsNet/FakeNewsNet_emotions.csv",
        "embedding" :"/features/FakeNewsNet/FakeNewsNet_embedding.csv"
        },
    "isot":
        {
        "lexicon"   : "/features/ISOT/ISOT_lexicon.csv",
        "semantic"  : "/features/ISOT/ISOT_sementic.csv",
        "emotion"   : "/features/ISOT/ISOT_emotions.csv",
        "embedding" : "/features/ISOT/ISOT_embedding.csv"
        },
    # "kaggle" :["","/features/Kaggle/Kaggle_sementic.csv","/features/Kaggle/Kaggle_predictions_emotions.csv"],
    "kagglerealfake":
        {
        "lexicon"  :"/features/Kaggle_real_fake/Kaggle_real_fake_lexicon.csv",
        "semantic" :"/features/Kaggle_real_fake/Kaggle_real_fake_sementic.csv",
        "emotion"  : "/features/Kaggle_real_fake/Kaggle_real_fake_emotions.csv",
        "embedding" : "/features/Kaggle_real_fake/Kaggle_real_fake_embedding.csv"
        },
    "liar":
        {
        "lexicon"   : "/features/LIAR/Liar_lexicon.csv",
        "semantic"  : "/features/LIAR/Liar_sementic.csv",
        "emotion"   : "/features/LIAR/LIAR_emotions.csv" ,
        "embedding" : "/features/LIAR/LIAR_embedding.csv"
        },
    # "politifact":["","/features/Politifact/Politifact_sementic.csv","/features/Politifact/Politifact_predictions_emotions.csv"],
    
    # "welfake":
    #     {
    #     "lexicon" :"/features/Welfake/Welfake_lexicon.csv",
    #     "semantic":"/features/Welfake/Welfake_sementic.csv",
    #     "emotion" : "/features/Welfake/WelFake_emotions.csv",
    #     "embedding":  "/features/Welfake/WelFake_embedding.csv"
    #     }
}
ID={"codalab":"id", "fakenewsnet":"id_1", "isot":"id", "kaggle":"id","kagglerealfake":"id", "liar":"ID", "politifact":"claim_id", "welfake":"id"}
LABEL={"codalab":"label", "fakenewsnet":"label", "isot":"label", "kaggle":"label","kagglerealfake":"label","liar":"label", "politifact":"cred_label", "welfake":"label"}
FEATURES=["lexicon","sementic","sentiment","embedding"]
TEXT={"codalab":"tweet", "fakenewsnet":"title", "isot":"text", "kaggle":"text","kagglerealfake":"text", "liar":"statement", "politifact":"text", "welfake":"text"}
