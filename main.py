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




