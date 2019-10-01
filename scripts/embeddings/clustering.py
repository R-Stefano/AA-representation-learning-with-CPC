import numpy as np 
from sklearn.mixture import GaussianMixture as GMM
import yaml
import pickle

with open('../../hyperparams.yml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']

embeddings=np.load(data_dir+'embeddings.npy', allow_pickle=True)

print(embeddings.shape)
gmm = GMM(n_components=4).fit(embeddings)

pickle.dump(gmm, open(data_dir+'models/GMM', 'wb'))