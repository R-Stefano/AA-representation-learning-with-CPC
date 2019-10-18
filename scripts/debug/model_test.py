import yaml
import sys
import numpy as np
import tensorflow as tf
with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']

sys.path.append(hyperparams['shared_scripts'])
import CPC as model_wrapper 

np.random.seed(0)
tf.random.set_seed(0)

batch_size=5
sequence_length=512

import h5py

train_dataset=h5py.File(data_dir+'dataset/unsupervised_large_clusters/train_dataset.hdf5', 'r')['sequences'][:batch_size]
print(train_dataset.shape)

model_utils=model_wrapper.Model(hyperparams['models_dir'],'CPC_untrained')
model=model_utils.architecture()
model_dir=model_utils.dir

#model_utils.exportModel(model)
#print(model.predict(inputData).shape)

train_generator=model_utils.BatchGenerator(train_dataset, batch_size)
test_generator=model_utils.BatchGenerator(train_dataset, batch_size)

#model_utils.exportModel(model)

model.fit_generator(
    generator=train_generator,
    validation_data=test_generator,
    epochs=10000,
    verbose=1
)
