import yaml
import sys
import numpy as np
import tensorflow as tf
with open('../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import CPC as model_wrapper #seq2seq

np.random.seed(0)
tf.random.set_seed(0)

batch_size=64
sequence_length=512

inputData=np.random.randint(0,20,size=(batch_size, sequence_length))
targetData=np.random.randint(0,20,size=(batch_size, sequence_length))


model_utils=model_wrapper.Model(hyperparams['models_dir'],'test')
model=model_utils.architecture()
model_dir=model_utils.dir

'''
model.load_weights(model_dir+'model.01-0.00.hdf5')
out=model.predict([inputData, targetData])

print(out.shape)
print(out)
'''
callbacks=[
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir+'model.{epoch:02d}-{val_loss:.2f}.hdf5' ,
        monitor='val_custom_accuracy', 
        load_weights_on_restart=True, 
        save_best_only=True),
]

train_generator=model_utils.BatchGenerator(inputData, batch_size)
test_generator=model_utils.BatchGenerator(inputData, batch_size)

model.fit_generator(
    generator=train_generator,
    validation_data=test_generator,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)