import numpy as np
import pickle
import yaml

dataset_file='../extract/sequences'
with open(dataset_file, 'rb') as f:
    dataset=pickle.load(f)

with open('../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

aminos_list=hyperparams['aminos']
sequence_length=hyperparams['max_sequence_length']

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

np.save('data/train_dataset.npy', train_set)
np.save('data/test_dataset.npy', test_set)



