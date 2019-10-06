import tensorflow as tf
import yaml
import sys
import numpy as np
import pickle
import pandas as pd

import query_API as api

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']

with open(data_dir+'dataset_config.yaml') as f:
    dataset_configs=yaml.safe_load(f)

tokes_voc=dataset_configs['aa_vocabulary']
max_seq_length=dataset_configs['sequence_length']

validation=np.load(data_dir+'dataset/secondary_structure/validation_sst8.npy')

models_dir=hyperparams['models_dir']
model_base_name='Transformer_untrained'
model_tuner_name='tuner_secondary'

model=tf.keras.models.load_model(models_dir+model_base_name+'/'+model_tuner_name+'/model')

labels=8 +1
batch_size=10
def evaluateLabels(y_true, y_preds):
    #remove prediction for 0
    y_preds=y_preds[:, :, 1:]

    #remove predictions where input was padding
    mask=y_true!=0
    y_true=y_true[mask]
    y_preds=y_preds[mask]

    #get model's predictions (increase by 1 to compensate the crop of pred 0)
    y_preds=np.argmax(y_preds, axis=-1)+1

    batch_results=[]
    #compute metrics for each label
    for label_idx in range(1, labels):
        label_mask=y_true==label_idx

        label_preds=y_preds[label_mask]

        tp=np.sum(label_preds==label_idx)
        fp=np.sum(y_preds[~label_mask]==label_idx)
        fn=np.sum(label_preds!=label_idx)

        batch_results.append([tp, fp, fn])

    return batch_results

evaluate_labels_results=[]
for b_start in range(0, validation.shape[1], batch_size):
    b_end=b_start+batch_size

    X=validation[0, b_start:b_end]
    Y=validation[1, b_start:b_end]

    preds=model.predict(X)

    batch_result=evaluateLabels(Y, preds)

    evaluate_labels_results.append(batch_result)

for l_idx in range(labels-1):
    label_res=np.asarray(evaluate_labels_results)[:, l_idx]

    tp=np.sum(label_res[:, 0])
    fp=np.sum(label_res[:, 1])
    fn=np.sum(label_res[:, 2])

    prec=tp/(tp+fp+1e-9)
    recall=tp/(tp+fn+1e-9)

    #print('Label {} | precision: {:.2f} | recall: {:.2f}'.format(labels_list[l_idx], prec, recall))
    print('Label {} | precision: {:.2f} | recall: {:.2f}'.format(l_idx, prec, recall))

'''
def prepareSequence(seq_string):
    print('Encoding sequence..')
    encoded_seq=[tokes_voc['<BOS>']]
    for aa in seq_string:
        encoded_seq.append(tokes_voc[aa])
    encoded_seq.append(tokes_voc['<EOS>'])

    while not len(encoded_seq)==max_seq_length:
        encoded_seq.append(tokes_voc['<PAD>'])
    
    return np.asarray(encoded_seq)

def rankingSimilarities(embedding):
    print('Ranking sequences..')
    from sklearn.metrics.pairwise import cosine_similarity

    rank=np.squeeze(cosine_similarity(embedding, dataset_embeddings))

    best_idx=np.argsort(rank)[-1]#idx 0 contains less similar prot
    score=rank[best_idx]

    return best_idx, score

def retrieveMostSimilarProteinInfo(best_idx):
    print('Retrieving most similar sequence..')
    #use the idx to get protein ID
    pdb_id=dataset_pdb_ids[best_idx]

    #Get Uniprot ID to retrieve infos
    uniprot_acc=api.getUniprotACC([pdb_id])

    print('Uniprot acc:', uniprot_acc)
    #Get data associated with best match
    data=api.getDataFromUniprot(uniprot_acc)

    return uniprot_acc, data

model=prepareModel(models_dir, model_name)

queries=pd.read_csv('query_input.csv')

results={
    'query_id':[],
    'uniprot_acc':[],
    'score':[]
    'data':[]
}

for i, q in enumerate(queries.iterrows()):
    id_uniprot=q[1]['uniprot_acc']
    print('Query: {} | {}'.format(i, id_uniprot))
    seq=q[1]['seq']


    inputData=prepareSequence(seq).reshape(1,-1)

    embedding=model.predict(inputData)[0, -1]

    best_idx, score=rankingSimilarities(embedding.reshape(1,-1))

    uniprot_acc, data=retrieveMostSimilarProteinInfo(best_idx)

    for u_acc in uniprot_acc:
        results['query_id'].append(i)
        results['score'].append(score)
        
        results['uniprot_acc'].append(u_acc)
        results['data'].append(data)



pd.DataFrame(results).to_csv('results.csv')
'''