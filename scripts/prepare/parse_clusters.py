import urllib.parse
import urllib.request
from itertools import islice
import pandas as pd
import yaml
import time
import os

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

data_dir=hyperparams['data_dir']
url = 'https://www.uniprot.org/uploadlists/'

def batchDataset(clusters_batch):
    '''
    First, extract the proteins id associated to each cluster id. Then, extract the 
    AA sequence for each protein
    Args:
        -clusters_batch (list): contains a list of uniref clusters id 
    '''
    cluster_entries={'cluster_ref':[], 'entry_id':[]}
    cluster_seqs={'entry_id':[], 'sequence':[]}

    #>>Extract proteins IDs from cluster IDs
    params = {
    'from': 'NF50',
    'to': 'ACC',
    'format': 'tab',
    'query': ' '.join(clusters_batch),
    'columns': 'id,protein names, id'
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    for _ in range(100):
        try:
            req = urllib.request.Request(url, data)

            with urllib.request.urlopen(req) as f:
                response = f.read()
                response=response.decode('utf-8')

                for idx, l in enumerate(response.split('\n')):
                    if idx==0 or len(l)==0: #skip first and last line
                        continue

                    line_split=l.split()
                    entry_id=line_split[0]
                    cluster_ref=line_split[-1]

                    cluster_entries['entry_id'].append(entry_id)
                    cluster_entries['cluster_ref'].append(cluster_ref)
            break
        except:
            print('No response, trying again to retrieve proteins id in clusters..')

    params = {
        'from': 'ACC',
        'to': 'ACC',
        'format': 'tab',
        'query': ' '.join(cluster_entries['entry_id']),
        'columns': 'sequence, feature(ACTIVE SITE), feature(BINDING SITE), feature(DNA BINDING), comment(CATALYTIC ACTIVITY)'
        }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')

    for _ in range(100):
        try:
            req = urllib.request.Request(url, data)

            with urllib.request.urlopen(req) as f:
                response = f.read()
                response=response.decode('utf-8')
                for idx, l in enumerate(response.split('\n')):
                    if idx==0 or len(l)==0: #skip first and last line
                        continue

                    line_split=l.split()
                    cluster_seqs['sequence'].append(line_split[0])
                    cluster_seqs['entry_id'].append(line_split[1])
        except:
            print('No response, trying again to retrieve proteins sequences..')

    cluster_seqs=pd.DataFrame(cluster_seqs)
    cluster_entries=pd.DataFrame(cluster_entries)
    dataset=cluster_entries.merge(cluster_seqs, on='entry_id', how='outer')

    return dataset

clusters_file=data_dir+'raw/clusters_uniref_50.txt'
destination_shard=data_dir+'raw/clusters/'
dataset=pd.DataFrame()

current_shard=len(os.listdir(destination_shard))

with open(clusters_file) as f:
    b_shard=0
    while True:
        next_n_lines = list(islice(f, 5000))

        #remove new line element
        clusters=[entry.strip() for entry in next_n_lines]
        
        if b_shard>=current_shard:

            time_s=time.time()
            batch=batchDataset(clusters)

            dataset=pd.concat([dataset, batch], ignore_index=True)

            print('Saving shard: {} in {:.3f}s'.format(b_shard, time.time()-time_s))
            dataset.to_csv(destination_shard+'shard_'+str(b_shard)+'.csv', index=False)
            time_s=time.time()
            dataset=pd.DataFrame()
        else:
            print('Skipping shard', b_shard)
        
        b_shard += 1

        if not next_n_lines:
            break