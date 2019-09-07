import tensorflow as tf
import yaml
import sys
import numpy as np

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import cpc as Model

models_dir=hyperparams['models_dir']

model_utils=Model.Model()

model_dir=models_dir+model_utils.name

custom_objects={
	'custom_xent':model_utils.custom_xent,
    'custom_accuracy': model_utils.custom_accuracy
}

print('>Loading model..')
model=tf.keras.models.load_model(model_dir+'model.h5', custom_objects=custom_objects)
#model.summary()

for layer in model.layers:
    print(layer.name)

'''
rnn_output=tf.keras.Model(
    inputs=model.get_layer('encoder_input').input,
    outputs=model.get_layer('rnn').output
)
'''