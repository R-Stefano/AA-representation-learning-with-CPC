import numpy as np
from annoy import AnnoyIndex
import lmdb
import yaml

import evaluate_embeddings_utils as utils

configs=yaml.safe_load(open('../../hyperparams.yml', 'r'))

data_dir=configs['data_dir']

token_voc=yaml.safe_load(open(data_dir+'dataset_config.yaml', 'r'))['aa_vocabulary']

embedding_size=512
u = AnnoyIndex(embedding_size, 'euclidean')
u.load(data_dir+'annoyIndex.ann')

env = lmdb.open(data_dir+'reference_id_DB.lmdb', map_size=int(1e9))

test_seq=['MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS']

embedded_seq=utils.encodeSeq(test_seq, token_voc, embedding_size, token_voc)
results=utils.queryEmbeddingsDB(embedded_seq, u)

for res in results:
    idxs, scores=res[0], res[1]
    similar_ids=utils.querySequenceRefDB(idxs, env)

    for sim_id, score in zip(similar_ids, scores):
        print(sim_id, score)
    
    break