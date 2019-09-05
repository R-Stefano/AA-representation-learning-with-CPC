import numpy as np
import pickle
import yaml

with open('../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']
aminos_list=hyperparams['aminos']
sequence_length=hyperparams['max_sequence_length']
np.random.seed(hyperparams['seed'])

raw_sequences='sequences'
with open(data_dir+raw_sequences, 'rb') as f:
    dataset=pickle.load(f)

encoded_dataset=[]
for idx, seq in enumerate(dataset):
    print(idx)
    encoded_seq=np.zeros((sequence_length), dtype=np.int8)-1
    if len(seq)<=sequence_length:
        for i, aa in enumerate(seq):
            encoded_seq[i]=aminos_list.index(aa)
        encoded_dataset.append(encoded_seq)

encoded_dataset=np.asarray(encoded_dataset)

dataset_size=len(encoded_dataset)
idxs=np.arange(dataset_size)
np.random.shuffle(idxs)

split_point=int(dataset_size*hyperparams['test_size'])
train_idxs=idxs[:-split_point]
test_idxs=idxs[-split_point:]

train_set, test_set=encoded_dataset[train_idxs], encoded_dataset[test_idxs]

print('Train dataset size:', len(train_set))
print('Test dataset size:', len(test_set))

np.save(data_dir+'train_dataset.npy', train_set)
np.save(data_dir+'test_dataset.npy', test_set)



