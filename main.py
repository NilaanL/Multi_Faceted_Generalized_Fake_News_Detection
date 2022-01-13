import functions.emotion_feature
import functions.sementic_feature
import functions.lexicon_creation
import functions.postprocessing
import functions.preprocessing
import functions
import config
from tqdm.auto import tqdm
import functions.helper

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tqdm.pandas()

feature_result_path = config.feature_result_path

id={"codalab":"id", "fakenewsnet":"id", "isot":"id", "kaggle":"id","kagglerealfake":"id", "liar":"ID", "politifact":"claim_id", "welfake":"id"}
label={"codalab":"label", "fakenewsnet":"label", "isot":"label", "kaggle":"label","kagglerealfake":"label","liar":"label", "politifact":"cred_label", "welfake":"label"}
features=["lexicon","sementic","sentiment"]

for key,value in feature_result_path.items():
    print("---------------",key,"---------------")
    error=False
    ID=id[key]
    Label=label[key]
    for v in range(3):
        if value[v]=="":
            error=True
            print("   Error: missing {:} skipping {:}".format(features[v],key))
            break
    if (not error):
        dfLexicon  = pd.read_csv(value[0])
        dfSementic = pd.read_csv(value[1])
        dfSentiment = pd.read_csv(value[2])

        dff=dfSentiment.merge(dfSementic, how='inner', on=ID,suffixes=('_Sentiment', '_Sementic'))
        df=dff.merge(dfLexicon, how='inner', on=ID,suffixes=('', '_Lexicon'))
        df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

        # df["qn_symbol_per_sentence"]=df["qn_symbol"]/df["num_sentences"]
        # df["num_exclamation_per_sentence"]=df["num_exclamation"]/df["num_sentences"]
        # df["url_count_per_sentence"]=df["url_count"]/df["num_sentences"]

        df=df.loc[df["lang"]=="en"]
        df["label"]=df[Label+"_Sementic"]

        print(df[Label].value_counts())
        # print(df.head())
        if (key=="politifact"):
            df["label"]=df["label"].replace(["true","mostly true","half-true"],0)
            df["label"]=df["label"].replace(["false","mostly false","pants on fire!"],1)

        if (key =="liar"):
            df["label"]=df["label"].replace(["true","mostly-true","half-true"],0)
            df["label"]=df["label"].replace(["false","barely-true","pants-fire"],1)
        
        if (key=="codalab" or key=="liar"):  
            df=df[All_features+["label",ID,"split_Sementic"]]

            print("null rows : ",df.isnull().any(axis=1).sum())
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            print("inf rows : ",df.isnull().any(axis=1).sum())
            df.dropna(inplace=True)

            df_train = df.loc[df["split_Sementic"]=="train"][All_features+['label']]
            df_test = df.loc[df["split_Sementic"]!="train"][All_features+['label']]  

            X_train=df_train[All_features]
            y_train=df_train["label"]
            X_val=df_test[All_features]
            y_val=df_test["label"]
        else:
            df=df[All_features+["label",ID,]]

            print("null rows : ",df.isnull().any(axis=1).sum())
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            print("inf rows : ",df.isnull().any(axis=1).sum())
            df.dropna(inplace=True)

            X_train, X_val, y_train, y_val = train_test_split(df[All_features], df["label"], test_size=0.3, random_state=142,stratify=df["label"])
        
        # print("   getting optimal Features :rfecv")
        # rfec_features=rfecv(X_train, X_val, y_train, y_val,key,resultsColumns,results)

        print("   comparing differnet models")
        compare_models( X_train, X_val, y_train, y_val ,All_features,key)
        print("=============================================================================")
        print("=============================================================================")


