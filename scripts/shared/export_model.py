import tensorflow as tf
import yaml
import sys

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']

sys.path.append(hyperparams['shared_scripts'])
import CPC as model_wrapper

model_name='CPC_v1'
base_dir=data_dir+'models/'+model_name+'/'
base_model_utils=model_wrapper.Model(data_dir+'models/', model_name)

model=base_model_utils.architecture()
model.load_weights(base_dir+'model_04.hdf5')

base_model_utils.exportModel(model)
