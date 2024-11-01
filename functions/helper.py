from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import torch
import config.py

def write_to_pickle(Pkl_File_path,model):
  with open(Pkl_File_path, 'wb') as file:  
      pickle.dump(model, file)

def read_pickle_model(path):
  with open(path, 'rb') as file:  
      return pickle.load(file)

def compute_metrics(pred,ground_labels):
    labels_all = ground_labels
    preds_all = list(pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all)
    acc = accuracy_score(labels_all, preds_all)
    confusion_mat = confusion_matrix(labels_all, preds_all)
    # tn, fp, fn, tp = confusiton_mat.ravel()
    out_dict = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusiton_mat': confusion_mat
      }
    return out_dict

# mapping of the labels to 0,1 
def label_map(x): 
  if x in ['true', 'mostly-true', 'half-true', 'real', 'Real', 0, 'REAL']:
    return 0
  elif x in ['false', 'pants-fire', 'barely-true', 'fake', 'Fake', 1, 'FAKE']:
    return 1
  else:return x


def normalize(dataFrame,features , parameterDict={}):
  dataframe=dataFrame.copy()
  for column in dataframe[features].columns.tolist():
    Q1=dataframe[column].quantile(0.25)
    Q3=dataframe[column].quantile(0.75)

    IQR=(Q3-Q1)
    minV=Q1 - 1.5*IQR
    maxV=Q3 + 1.5*IQR


    temp=dataframe[column].copy()
  
    if ( column not in ["qn_symbol_per_sentence" , "num_exclamation_per_sentence" ,"lexical_diversity" ,"url_count_per_sentence"] ) :
      dataframe[column]=dataframe[column].apply(lambda x:minV if x< minV else maxV if x>maxV else x)

      mean = dataframe[column].mean()
      std  = dataframe[column].std() 

      try:
        dataframe[column]=dataframe[column].apply(lambda x:  (x-mean)/std )
      except:
        print(column) 

    else:
      dataframe[column]=dataframe[column].apply(lambda x : 1 if x>0 else 0)
      # print("col",column)
  return dataframe



def rfecv(X_train, X_val, y_train, y_val,key,results):
    clf=RandomForestClassifier(n_estimators=100,max_features='auto',random_state=0,max_depth=14,class_weight='balanced')
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(),scoring='accuracy')
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    plt.figure(figsize=(5,5))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    print(X_train.columns.values[rfecv.support_])
    rfecv_features=list(X_train.columns.values[rfecv.support_])
    print(len(rfecv_features)," : " ,rfecv_features)

    predicted_y = rfecv.predict(X_val)
    gg=compute_metrics(predicted_y,y_val)
    d=gg
    tn, fp, fn, tp = d["confusiton_mat"].ravel()
    # tn, fp, fn, tp = 0,0,0,0
    print("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format('prec-t','prec-f', 'rec-t','rec-f','f1-t','f1-f','accu','tn', 'fp', 'fn', 'tp'))  # correct
    print ("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:}\t{:}\t{:}\t{:}".format(d['precision'][0],d['precision'][1], d['recall'][0],d['recall'][1],d['f1'][0],d['f1'][1],d['accuracy'],tn, fp, fn, tp))

    # dfrfecv=pd.DataFrame([[key,"rfecv",d['precision'][0],d['precision'][1], d['recall'][0],d['recall'][1],d['f1'][0],d['f1'][1],d['accuracy'],tn, fp, fn, tp]],columns=resultsColumns)
    dfrfecv=[key,"rfecv",d['precision'][0],d['precision'][1], d['recall'][0],d['recall'][1],d['f1'][0],d['f1'][1],d['accuracy'],tn, fp, fn, tp]
    results.append(dfrfecv)
    return(rfecv_features)

def compare_models( X_train, X_val, y_train, y_val,features,dataset_name,classifiers,write_nodel_to_file=False,model_export_path=None):
    """compare different models and print accuracy score

    Args:
        X_train (Pandas.Dataframe): Taining features
        X_val (Pandas.Dataframe): Validation features
        y_train (Pandas.Dataframe): Training labels
        y_val (Pandas.Dataframe): Validation labels
        features (Pandas.Dataframe): Features to be used
        key (Pandas.Dataframe): 
        classifiers ({<String>:<Modal>}): Dictionary of classifiers to be used for comparison {key<String>:value<Modal>}
        write_nodel_to_file (bool, optional): . Defaults to False.
        model_export_path (String, optional): Root path for model export. Defaults to None.

    Returns:
        [List]: List of results
    """
    cla_pred=[]

    for name,model1 in classifiers:
        print("-----------"+name+"-------------")
        model1.fit(X_train[features],y_train)
        predicted_y = model1.predict(X_val[features])
        score=compute_metrics(predicted_y,y_val)
        cla_pred.append(score)
        if write_nodel_to_file: 
            write_to_pickle(model_export_path+key+"_"+name+".pkl",model1)
      
    #Prining the evaluation matrix to the console
    print("Summary\n Class 0- True news \n Class 1 - False news\n {:}".format(dataset_name))
    print("{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format('prec-t','prec-f', 'rec-t','rec-f','f1-t','f1-f','accu','tn', 'fp', 'fn', 'tp',"model"))  # correct
    for i in range(len(classifiers)):
        d=cla_pred[i]
        tn, fp, fn, tp = cla_pred[i]["confusiton_mat"].ravel() #correct
        print ("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:}\t{:}\t{:}\t{:}\t{:}".format(d['precision'][0],d['precision'][1], d['recall'][0],d['recall'][1],d['f1'][0],d['f1'][1],d['accuracy'],tn, fp, fn, tp,classifiers[i][0]))

    return cla_pred

def generate_embeddings(model,sentences):
    """generate embeddings for a list of sentences

    Args:
        sentences (list): List of sentences

    Returns:
        [list]: List of embeddings
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)   ##to run on gpu
    embeddings = model.encode(sentences=sentences, show_progress_bar=True)
    embeddings=[torch.from_numpy(item) for item in embeddings]
    
    return embeddings

def convert_to_array(tensor):
  return tensor.detach().cpu().numpy().tolist()

def get_tokens(text):
  return len(re.findall(r'\w+', text))

def get_train_valid_test_split(dataframe,key,columns,train_size=0.7,valid_size=0.2,test_size=0.1):
    """split dataframe into train,validation and test sets

    Args:
        dataframe (Pandas.DataFrame): [description]
        key ([type]): [description]
        columns ([type]): [description]
        train_size (float, optional): [description]. Defaults to 0.7.
        valid_size (float, optional): [description]. Defaults to 0.2.
        test_size (float, optional): [description]. Defaults to 0.1.
    Returns:
        [list]: List of train,validation and test dataframes
    """
 
    dff=dataframe
    if key=="liar":
        train = dff.loc[dff["split_Sementic"]=="train"]
        valid = dff.loc[dff["split_Sementic"]=="valid"]
        test = dff.loc[dff["split_Sementic"]=="test"]

        X_train_1 = train[columns]
        y_train_1 =train["label"]

        X_valid_1 = valid[columns]
        y_valid_1 =valid["label"]

        X_test_1 =test[columns]
        y_test_1 =test["label"]
    elif key=="codalab":
        train = dff.loc[dff["split_Sementic"]=="train"]
        valid = dff.loc[dff["split_Sementic"]=="val"]
        test = dff.loc[dff["split_Sementic"]=="test"]

        X_train_1 = train[columns]
        y_train_1 =train["label"]

        X_valid_1 = valid[columns]
        y_valid_1 =valid["label"]

        X_test_1 =test[columns]
        y_test_1 =test["label"]

    else:
        X_train_1, y_train_1, X_valid_1, y_valid_1, X_test_1, y_test_1 = train_valid_test_split(dff[columns+["label"]], target = 'label', train_size, valid_size, test_size)

    X_train_1 = X_train_1.to_numpy().tolist()
    y_train_1 = y_train_1.to_numpy().tolist()

    X_valid_1 = X_valid_1.to_numpy().tolist()
    y_valid_1 = y_valid_1.to_numpy().tolist()

    X_test_1 = X_test_1.to_numpy().tolist()
    y_test_1 = y_test_1.to_numpy().tolist()

    for i in range (len(X_train_1)):
        list_str = str(X_train_1[i]).replace("[", "").replace("]", "")
        X_train_1[i] = eval(f"[{list_str}]")
    X_train_1 = np.array(X_train_1)
    y_train_1 = np.array(y_train_1)

    for i in range (len(X_valid_1)):
        list_str = str(X_valid_1[i]).replace("[", "").replace("]", "")
        X_valid_1[i] = eval(f"[{list_str}]")
    X_valid_1 = np.array(X_valid_1)
    y_valid_1 = np.array(y_valid_1)

    for i in range (len(X_test_1)):
        list_str = str(X_test_1[i]).replace("[", "").replace("]", "")
        X_test_1[i] = eval(f"[{list_str}]")
    X_test_1 = np.array(X_test_1)
    y_test_1 = np.array(y_test_1)

    return (X_train_1,X_valid_1,X_test_1,y_train_1,y_valid_1,y_test_1)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def predict_for_model(X_val, y_val,features,dataset_name,model,model_name): 
  # print("-----------"+model_name+"-------------")
    predicted_y = model.predict(X_val[features])
    score=compute_metrics(predicted_y,y_val)

    #Prining the evaluation matrix to the console
    d=score
    predictionScore.append(d)
    tn, fp, fn, tp = d["confusiton_mat"].ravel() #correct
    print ("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:}\t{:}\t{:}\t{:}\t{:}\t\t\t{:}".format(d['precision'][0],d['precision'][1], d['recall'][0],d['recall'][1],d['f1'][0],d['f1'][1],d['accuracy'],tn, fp, fn, tp,dataset_name,model_name))

def get_all_df_dict(normalize=False ,normalize_feature_list=[])
    all_df={}

    # for each dataset 
    for key,value in feature_result_path.items():
        print("----------",key,"--------------")
        error=False
        ID=ID[key]
        TEXT=TEXT[key]
        LABEL=LABEL[key]

        # check whether all 4 features are available. 
        
        for feature_name,feature_dataset_location in value.items():
            if feature_dataset_location =="":
                error=True
                print("   Error: missing {:} skipping {:}".format(features[v],key))
                break
        if (not error):
            # read each of the 4 features for the dataset 
            dfLexicon  = pd.read_csv(value["lexicon"])
            dfSementic = pd.read_csv(value["semantic"])
            dfSentiment = pd.read_csv(value["emotion"])
            dfEmbedding = pd.read_csv(value["embedding"])

            # combine the features using inner join 

            dff=dfSentiment.merge(dfSementic, how='inner', on=ID,suffixes=('_Sentiment', '_Sementic'))
            dff=dff.merge(dfLexicon, how='inner', on=ID,suffixes=('', '_Lexicon'))
            df=dff.merge(dfEmbedding, how='inner', on=ID,suffixes=('', '_Lexicon'))

            df=df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
        
            df["scores"]=df["scores"].apply(lambda x : list(map(np.float64, x.strip('][').replace('"', '').replace("'","").replace(" ","").split(','))))
            # df["fake_score"]=df["scores"].apply(lambda x:x[0])
            # df["true_score"]=df["scores"].apply(lambda x:x[1])

            df["fake_score1"]=df["scores"].apply(lambda x:x[2])
            df["true_score1"]=df["scores"].apply(lambda x:x[3])

            df["fake_score2"]=df["scores"].apply(lambda x:x[4])
            df["true_score2"]=df["scores"].apply(lambda x:x[5])

            df=df.loc[df["lang"]=="en"]    # filter only english text 
            df["label"]=df[LABEL+"_Sementic"]  #set the label coulmn 

            # print(df["label"].value_counts())

            df["label"]=df["label"].apply(label_map)   # converting labels to 0,1 
        
            
            # for codalab and liar include split column as well 
            if (key=="codalab" or key=="liar"):  
                df=df[All_features+["label",ID,"split_Sementic"]]
            else :
                df=df[All_features+["label",ID,]]
            
            #clean 
            print("null rows : ",df.isnull().any(axis=1).sum())
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            print("inf rows : ",df.isnull().any(axis=1).sum())
            df.dropna(inplace=True)
        
            if (NORMALIZED):
                all_df[key]=normalize(df,All_features)   # normalize 
            else:
                all_df[key]=df.copy(deep=True)        
    return all_df