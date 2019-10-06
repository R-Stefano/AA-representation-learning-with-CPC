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

dataset='sst3'
labels_voc=list(dataset_configs['secondary_task'][dataset].keys())[1:]
print(labels_voc)
labels=len(labels_voc)
batch_size=10

validation=np.load(data_dir+'dataset/secondary_structure/validating_'+dataset+'.npy')

models_dir=hyperparams['models_dir']
model_base_name='Transformer_untrained'
model_tuner_name='tuner_secondary_'+dataset

model=tf.keras.models.load_model(models_dir+model_base_name+'/'+model_tuner_name+'/model')

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
    for label_idx in range(1, labels+1):
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

for l_idx in range(labels):
    label_res=np.asarray(evaluate_labels_results)[:, l_idx]

    tp=np.sum(label_res[:, 0])
    fp=np.sum(label_res[:, 1])
    fn=np.sum(label_res[:, 2])

    prec=tp/(tp+fp+1e-9)
    recall=tp/(tp+fn+1e-9)

    print('Label {} | precision: {:.2f} | recall: {:.2f}'.format(labels_voc[l_idx], prec, recall))