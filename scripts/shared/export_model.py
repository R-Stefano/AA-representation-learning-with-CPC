import tensorflow as tf
import yaml
import sys

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']

sys.path.append(hyperparams['shared_scripts'])
import tuner as model_wrapper

model_name='CPC_v1'
base_dir=data_dir+'models/'+model_name+'/tuner_secondary_sst3/'
base_model_utils=model_wrapper.Model(data_dir+'models/', model_name)
#base_model_utils=model_wrapper.Model(data_dir+'models/', model_name, 'test')

model=base_model_utils.architecture()
model.load_weights(base_dir+'model_21.hdf5')

base_model_utils.exportModel(model)
