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
    "codalab": "/datasets/CodaLab Covid/Constraint_English_All.csv",
    "fakenewsnet":"/datasets/FakeNewsNet/FakeNewsNet_All.csv",
    "isot":"/datasets/ISOT/ISOT.csv",
    # "kaggle":"/datasets/Kaggle/Kaggle.csv",
    "kagglerealfake":"/datasets/Kaggle_real_fake/fake_or_real_news.csv",    
    "liar": "/datasets/LIAR/Liar_all.csv",
    "politifact":"/datasets/Politifact/Politifact.tsv",
    "welfake":"/datasets/WELFAKE/WELFake.csv"
}

feature_result_path={
    "codalab":
        [
        "/features/CodaLab Covid/CodaLab_lexicon.csv",
        "/features/CodaLab Covid/CodaLab_sementic.csv",
        "/features/CodaLab Covid/CodaLab_emotions.csv"
        ],
    "fakenewsnet" :
        [
        "/features/FakeNewsNet/FakeNewsNet_lexicon.csv",
        "/features/FakeNewsNet/FakeNewsNet_sementic.csv",
        "/features/FakeNewsNet/FakeNewsNet_emotions.csv"
        ],
    "isot":
        [
        "/features/ISOT/ISOT_lexicon.csv",
        "/features/ISOT/ISOT_sementic.csv",
        "/features/ISOT/ISOT_emotions.csv"
        ],
    # "kaggle" :["","/features/Kaggle/Kaggle_sementic.csv","/features/Kaggle/Kaggle_predictions_emotions.csv"],
    "kagglerealfake":
        [
        "/features/Kaggle_real_fake/Kaggle_real_fake_lexicon.csv",
        "/features/Kaggle_real_fake/Kaggle_real_fake_sementic.csv",
        "/features/Kaggle_real_fake/Kaggle_real_fake_emotions.csv"
        ],
    "liar":
        ["/features/LIAR/Liar_lexicon.csv",
        "/features/LIAR/Liar_sementic.csv",
        "/features/LIAR/LIAR_emotions.csv"
        ],
    # "politifact":["","/features/Politifact/Politifact_sementic.csv","/features/Politifact/Politifact_predictions_emotions.csv"],
    "welfake":
        [
        "/features/Welfake/Welfake_lexicon.csv",
        "/features/Welfake/Welfake_sementic.csv",
        "/features/Welfake/WelFake_emotions.csv"
        ]
}
