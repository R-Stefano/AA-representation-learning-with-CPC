import tensorflow as tf
import yaml
import sys

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']

sys.path.append(hyperparams['shared_scripts'])
import Transformer as model_wrapper

base_dir=data_dir+'models/Transformer_6/'
base_model_utils=model_wrapper.Model(data_dir+'models/', 'Transformer_6')

model=base_model_utils.architecture()
model.load_weights(base_dir+'model_01.hdf5')

base_model_utils.exportModel(model)
