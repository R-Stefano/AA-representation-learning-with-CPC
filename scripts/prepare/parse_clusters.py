import urllib.parse
import urllib.request
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

    params = {
        'from': 'ACC',
        'to': 'ACC',
        'format': 'tab',
        'query': ' '.join(cluster_entries['entry_id']),
        'columns': 'sequence, feature(ACTIVE SITE), feature(BINDING SITE), feature(DNA BINDING), comment(CATALYTIC ACTIVITY)'
        }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
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

    cluster_seqs=pd.DataFrame(cluster_seqs)
    cluster_entries=pd.DataFrame(cluster_entries)
    dataset=cluster_entries.merge(cluster_seqs, on='entry_id', how='outer')

    return dataset

clusters_file=data_dir+'raw/clusters_uniref_50.txt'

batch_size=5000
shard_size=100000
dataset=pd.DataFrame(columns=['cluster_ref', 'entry_id', 'sequence'])
clusters=[]

#Compute last cluster shard processed
processed_shard=0
for filename in os.listdir(data_dir+'raw/clusters/'):
    start_idx=filename.rfind('_')
    end_idx=filename.index('.')
    shard_number=filename[start_idx+1: end_idx]
    processed_shard=int(shard_number)

with open(clusters_file, 'r') as f:
    time_s=time.time()
    for idx, line in enumerate(f):

        #Start after the last cluster saved
        if (idx)//shard_size>processed_shard:
            clusters.append(line.strip())

            #Query uniref API
            if len(clusters)==batch_size:
                batch=batchDataset(clusters)
                dataset=dataset.append(batch, ignore_index=True)
                clusters=[]
                print('Batch ({}/{}) in {:.2f}s'.format(idx, '?', time.time()-time_s))
                time_s=time.time()

            #Save on disk to free memory
            if (idx+1)%shard_size==0:
                print('Generating shard:', idx//shard_size)
                batch=batchDataset(clusters)
                dataset=dataset.append(batch, ignore_index=True)
                clusters=[]
                #add cluster id
                cluster_ids=pd.DataFrame({'cluster_ref': dataset['cluster_ref'].unique()})
                cluster_ids['cluster_id']=cluster_ids.index.values

                dataset=dataset.merge(cluster_ids, on='cluster_ref', how='left')

                dataset.to_csv(data_dir+'raw/clusters/dataset_uniref_50_'+str(idx//shard_size)+'.csv')
                dataset=pd.DataFrame(columns=['cluster_ref', 'entry_id', 'sequence'])

batch=batchDataset(clusters)
dataset=dataset.append(batch, ignore_index=True)

#add cluster id
cluster_ids=pd.DataFrame({'cluster_ref': dataset['cluster_ref'].unique()})
cluster_ids['cluster_id']=cluster_ids.index.values

dataset=dataset.merge(cluster_ids, on='cluster_ref', how='left')

dataset.to_csv(data_dir+'raw/clusters/dataset_uniref_50_last.csv')