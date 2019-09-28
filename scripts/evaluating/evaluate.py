import tensorflow as tf
import yaml
import sys
import numpy as np

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import CPC as model_wrapper

models_dir=hyperparams['models_dir']
model_name='CPC_2' #CPC, CPC_2

def prepareModel(models_dir, model_name):
    model_utils=model_wrapper.Model(models_dir, model_name)
    model=model_utils.architecture()


    custom_objects={
        'custom_loss':model_utils.custom_loss,
        'custom_accuracy': model_utils.custom_accuracy
    }

    print('>Loading {} model'.format(model_name))
    model=tf.keras.models.load_model(model_dir+'model.h5', custom_objects=custom_objects)
    '''
    model.summary()
    for layer in model.layers:
        print(layer.name)
    '''

    rnn_output=tf.keras.Model(
        inputs=model.get_layer('encoder_input').input,
        outputs=model.get_layer('rnn').output
    )
    return rnn_output

model=prepareModel(models_dir, model_name)