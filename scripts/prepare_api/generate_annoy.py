import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
import h5py

from annoy import AnnoyIndex
import lmdb

batch_size=32
embedding_size=512
n_tree=30 # More trees gives higher precision when querying. obviously get slower

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']
dataset=h5py.File(data_dir+'dataset/unsupervised_large_clusters/dataset.hdf5')['sequences']
datatset_seq_ids=pd.read_csv(data_dir+'dataset/unsupervised_large_clusters/sequences_ids.csv', chunksize=batch_size)

model=tf.keras.models.load_model(data_dir+'models/Transformer_1/model/')

t = AnnoyIndex(embedding_size, 'euclidean')
idDB = lmdb.open(data_dir+'reference_id_DB.lmdb', map_size=int(1e9))

with idDB.begin(write=True) as idDB_write:
    for b_start, batch_ids in zip(range(0, len(dataset), batch_size), datatset_seq_ids):
        b_end=b_start+batch_size

        batch_data=dataset[b_start:b_end]

        #outs=model.predict(batch_data)
        #embeddings=np.mean(outs, axis=-1)

        embeddings=batch_data/23

        for idx, embed in enumerate(embeddings):
            seq_id=batch_ids.iloc[idx]['sequence_id'].encode()

            #keep unique index across batches
            idx+=b_start

            t.add_item(idx, embed)

            id = str(idx).encode()
            idDB_write.put(id, seq_id)
            idDB_write.put(seq_id, id)

t.build(n_tree)
t.save(data_dir+'annoyIndex.ann')