import yaml
import sys
import numpy as np

with open('../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import cpc as cpc_model

batch=3
timesteps=32
length=1

num_samples=4
window_size=34
code_size =128
rnn_units =256
num_predic_terms= 4
learning_rate=0.0001

inputData=np.random.normal(size=(batch, timesteps, window_size,length))
targetData=np.random.normal(size=(batch, timesteps, num_predic_terms, num_samples, window_size, length))

labels=np.zeros((batch*timesteps*num_predic_terms, num_samples))
idxs_y=np.random.randint(0, num_samples, size=(batch*timesteps*num_predic_terms))
idxs_x=np.arange(0, batch*timesteps*num_predic_terms)
labels[idxs_x, idxs_y]=1
labels=np.reshape(labels, (batch, timesteps, num_predic_terms, num_samples))

model_utils=cpc_model.Model()
model=model_utils.architecture(timesteps, num_predic_terms, num_samples, window_size, length, code_size, rnn_units, learning_rate)


model.fit(x=[inputData, targetData], y=labels, batch_size=batch, epochs=50)

preds=model.predict([inputData, targetData])
print(preds.shape)
print(np.min(preds), np.max(preds))