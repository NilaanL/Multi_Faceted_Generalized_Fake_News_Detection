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

