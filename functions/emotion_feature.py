import operator
from annoy import AnnoyIndex
import ast

def get_nearest_neighbours(embedding,total_text,t,lexicon_words,lexicon_emotions,lexicon_embeddings,lexicon_scores):
    # t1 = datetime.now()
    tuples = []
    
    embeding = embedding
    sentence_tokens = total_text.split(' ')

    # for i,row_e in df.iterrows():
        
    #     dis = cosine_similarity([row_e['embedding']], [embeding])
    #     # print([row_e['tokens'],row_d['tokens'],dis])
    #     tuples.append([row_e['word'],row_e['emotion'],dis,row_e['embedding'],row_e['emotion-intensity-score']])

    indexes, distances = t.get_nns_by_vector(embeding, 50, include_distances=True)
    for i in range(len(indexes)):
        tuples.append([lexicon_words[indexes[i]],lexicon_emotions[indexes[i]],distances[i],lexicon_embeddings[indexes[i]],lexicon_scores[indexes[i]]])
    # print(indexes)
    # print(distances)
    # print(tuples)
    
    s_tup = sorted(tuples, key=lambda x: x[2])#sort tuples based on the cosine distance
    neaarest_neighbs_words = []
    neaarest_neighbs_embs = []
    neaarest_neighbs_labels = []
    neaarest_neighbs_distance = []
    # neaarest_neighbs_positive = []
    # neaarest_neighbs_negative = []
    neaarest_neighbs_intensity_score = []
    for i,m in enumerate(s_tup[::-1]):
        # print(m)
        if(i<50):#getting the nearest 50 neighbours
            neaarest_neighbs_words.append(m[0])
            neaarest_neighbs_embs.append(m[3])
            neaarest_neighbs_labels.append(m[1])
            # distance = m[2].tolist()[0][0]
            distance = m[2]
            neaarest_neighbs_distance.append(distance)
            neaarest_neighbs_intensity_score.append(m[4])
            # neaarest_neighbs_positive.append(m[5])
            # neaarest_neighbs_negative.append(m[6])

    n_score_dict = calculate_scores(neaarest_neighbs_words,neaarest_neighbs_labels,neaarest_neighbs_distance,neaarest_neighbs_intensity_score
                                    # neaarest_neighbs_positive,neaarest_neighbs_negative
                                    )
    
    neighbour_output = [n_score_dict,{'words':neaarest_neighbs_words,'embs':neaarest_neighbs_embs,'labels':neaarest_neighbs_labels}]
    normalized_score_dict = neighbour_output[0]
  
    for key in normalized_score_dict:
        normalized_score_dict[key] = round((normalized_score_dict[key]/len(sentence_tokens)),4)

    normalized_score_dict = {k: round(v,3) for k, v in sorted(normalized_score_dict.items(), key=lambda item: item[1])}
    # print(normalized_score_dict)
    return normalized_score_dict



def calculate_scores(neaarest_neighbs_words,neaarest_neighbs_labels,neaarest_neighbs_distance,neaarest_neighbs_intensity_score):
    score_dict = {
                'anticipation':0,
                'anger':0,
                'fear':0,
                'sadness':0,
                'trust':0,
                'joy':0,
                'surprise':0,
                'disgust':0
                # 'positive':0,
                # 'negative':0
                }

    #Scoring Mechanism
    for i in range(0,len(neaarest_neighbs_words)):
        distance = 0
        if (neaarest_neighbs_distance[i] == 0) :
            distance = 1
        else :
            distance = (1/neaarest_neighbs_distance[i])
        
        score = (distance*(neaarest_neighbs_intensity_score[i]))  
        score_dict[neaarest_neighbs_labels[i]]=score_dict[neaarest_neighbs_labels[i]]+score

    # score_dict['positive']=score_dict['anger']+ score_dict['fear']+score_dict['disgust']+score_dict['sadness']
    # score_dict['negative']=score_dict['joy']+score_dict['trust']+score_dict['anticipation']+score_dict['surprise']
    #Normalising Mechanism
    normalized_score_dict = score_dict.copy()
    # for k in score_dict.keys():
    #   if score_dict[k] ==0:
    #     continue
    #     # del normalized_score_dict[k]
    #   else:
    #     normalized_score_dict[k]
    #     # normalized_score_dict[k] = round((score_dict[k]/score_max),3)

    return normalized_score_dict


def dict_to_result(emotion_dict):
    # del emotion_dict['positive']
    # # del emotion_dict['negative']
    if (all(value == 0 for value in emotion_dict.values())):
        highest_8 = None
    else:
        highest_8 = max(emotion_dict.items(), key=operator.itemgetter(1))[0]
    return highest_8

# result = emotion_candidates_recognition('That was good',1)
# x = dict_to_result(result)

def change_string_dict(emotion_str):
  res = ast.literal_eval(emotion_str)
  return res


def load_lexicon_embedding(embedding,graph_path):
    embedding = [ast.literal_eval(i) for i in embedding.values.tolist()]
    f = 768
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed

    # super fast, will just mmap the file
    t.load(graph_path)

    return t
