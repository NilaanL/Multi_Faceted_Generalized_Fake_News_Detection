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
#   print(dataframe.columns)
#   columns=['fake_score', 'true_score', 'common_score',
#               'joy', 'fear', 'disgust', 'anticipation', 'anger', 'sadness', 'surprise', 'trust',
#               'url_count', 'qn_symbol', 'num_chars', 'num_words', 'num_sentences', 'words_per_sentence', 'characters_per_word', 
#               'punctuations_per_sentence', 'num_exclamation', 'get_sentiment_polarity', 'lexical_diversity', 'content_word_diversity',
#               'redundancy', 'noun', 'verb', 'adj', 'adv', 'qn_symbol_per_sentence', 'num_exclamation_per_sentence',
#               'url_count_per_sentence', 'embedding']
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