import numpy as np
import os
import yaml
import sys
import tensorflow as tf
import pandas as pd 

with open('../../hyperparams.yml') as f:
    configs=yaml.safe_load(f)

models_dir=configs['models_dir']
model_name='CPC' #CPC, CPC_2
data_dir=configs['data_dir']

with open(data_dir+'dataset_config.yaml') as f:
    dataset_configs=yaml.safe_load(f)

tokes_voc=dataset_configs['aa_vocabulary']

sys.path.append(configs['shared_scripts'])
import CPC as model_wrapper


dataset_seqs=[]
dataset_pdb_ids=[]
for dataset_name in os.listdir(data_dir+'raw/csv/'):
    data=pd.read_csv(data_dir+'raw/csv/'+dataset_name)
    
    #remove entries too long
    data['len']=data['seqs'].str.len()
    data=data[data['len']<=dataset_configs['sequence_length']-2]

    #encode entry and store id
    for entry in data.iterrows():
        pdb_id=entry[1]['pdb_id']
        seq=entry[1]['seqs']

        #Extract pdb_id
        dataset_pdb_ids.append(pdb_id)

        #Encode sequence
        encoded_seq=[tokes_voc['<BOS>']]
        for aa in seq:
            encoded_seq.append(tokes_voc[aa])

        encoded_seq.append(tokes_voc['<EOS>'])

        while not len(encoded_seq)==dataset_configs['sequence_length']:
            encoded_seq.append(tokes_voc['<PAD>'])

        dataset_seqs.append(encoded_seq)

dataset_seqs=np.stack(dataset_seqs)
batch_size=500

model_utils=model_wrapper.Model(models_dir, model_name)
model=model_utils.architecture()

custom_objects={
    'custom_loss':model_utils.custom_loss,
    'custom_accuracy': model_utils.custom_accuracy
}

print('>Loading {} model'.format(model_name))
model=tf.keras.models.load_model(models_dir+model_name+'/model.01-1.98.hdf5', custom_objects=custom_objects)

model=tf.keras.Model(
    inputs=model.get_layer('encoder_input').input,
    outputs=model.get_layer('rnn').output
)

embeddings=[]
for b_start in range(0, len(dataset_seqs), batch_size):
    print('Batch ({}/{})'.format(b_start//batch_size, len(dataset_seqs)//batch_size))
    b_end=b_start+batch_size
    batch_data=dataset_seqs[b_start:b_end]
    out=model.predict(batch_data)
    embeddings.extend(out[:, -1])

np.save(data_dir+'embeddings.npy', embeddings)
np.save(data_dir+'pdb_ids.npy', dataset_pdb_ids)
