
def prepareSequence(seq_string):
    print('Encoding sequence..')
    encoded_seq=[tokes_voc['<BOS>']]
    for aa in seq_string:
        encoded_seq.append(tokes_voc[aa])
    encoded_seq.append(tokes_voc['<EOS>'])

    while not len(encoded_seq)==max_seq_length:
        encoded_seq.append(tokes_voc['<PAD>'])
    
    return np.asarray(encoded_seq)

def rankingSimilarities(embedding):
    print('Ranking sequences..')
    from sklearn.metrics.pairwise import cosine_similarity

    rank=np.squeeze(cosine_similarity(embedding, dataset_embeddings))

    best_idx=np.argsort(rank)[-1]#idx 0 contains less similar prot
    score=rank[best_idx]

    return best_idx, score

def retrieveMostSimilarProteinInfo(best_idx):
    print('Retrieving most similar sequence..')
    #use the idx to get protein ID
    pdb_id=dataset_pdb_ids[best_idx]

    #Get Uniprot ID to retrieve infos
    uniprot_acc=api.getUniprotACC([pdb_id])

    print('Uniprot acc:', uniprot_acc)
    #Get data associated with best match
    data=api.getDataFromUniprot(uniprot_acc)

    return uniprot_acc, data

model=prepareModel(models_dir, model_name)

queries=pd.read_csv('query_input.csv')

results={
    'query_id':[],
    'uniprot_acc':[],
    'score':[]
    'data':[]
}

for i, q in enumerate(queries.iterrows()):
    id_uniprot=q[1]['uniprot_acc']
    print('Query: {} | {}'.format(i, id_uniprot))
    seq=q[1]['seq']


    inputData=prepareSequence(seq).reshape(1,-1)

    embedding=model.predict(inputData)[0, -1]

    best_idx, score=rankingSimilarities(embedding.reshape(1,-1))

    uniprot_acc, data=retrieveMostSimilarProteinInfo(best_idx)

    for u_acc in uniprot_acc:
        results['query_id'].append(i)
        results['score'].append(score)
        
        results['uniprot_acc'].append(u_acc)
        results['data'].append(data)



pd.DataFrame(results).to_csv('results.csv')
