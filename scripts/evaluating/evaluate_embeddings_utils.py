import urllib.parse
import urllib.request

url = 'https://www.uniprot.org/uploadlists/'

def encodeSeq(seq_str, token_voc, embedding_size, token_voc):
    '''
    This function takes as input a list of AA sequences and returns a list of AA encoded.
    The encoding is done based on the vocabulary passed as parameter.

    Args:
        -seq_str (list): List of sequences uploaded. Each sequence comes as string of AA

    Returns:
        an array of encoded amino acids with shape [batch, 512]
    '''
    batch_size=len(seq_str)
    encoded_seq=np.zeros((batch_size, embedding_size), dtype=np.int8)

    for seq_idx, seq in enumerate(seq_str):
        encoded_seq[seq_idx][0]=token_voc['<BOS>']
        for aa_idx, aa  in enumerate(seq):
            aa_idx += 1

            if aa not in token_voc:
                token=token_voc['X']
            else:
                token=token_voc[aa]

            encoded_seq[seq_idx][aa_idx]=token

        encoded_seq[seq_idx][aa_idx+1]=token_voc['<EOS>']

    return encoded_seq/23

def queryEmbeddingsDB(query, u, n=10):
    '''
    Run search using annoy to find best N candidates for each query vector (embedding).
    It returns id of top N embeddings.

    '''
    results=[]
    for q in query:
        results.append(u.get_nns_by_vector(q, n, include_distances=True))

    return results

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

def querySequenceRefDB(elements_ids, env):
    similar_ids=[]
    with env.begin() as txn:
        for el_id in elements_ids:
            res=txn.get(str(el_id).encode()).decode()

            similar_ids.append(res)
    return similar_ids


def getDataFromUniprot(uniprot_accs):
   params = {
    'from': 'ACC',
    'to': 'ACC',
    'format': 'tab',
    'query': ' '.join(uniprot_accs),
    'columns': 'families'
    }

   data = urllib.parse.urlencode(params)
   data = data.encode('utf-8')
   req = urllib.request.Request(url, data)

   with urllib.request.urlopen(req) as f:
      response = f.read()
      response=response.decode('utf-8')
      for l in response.split('\n'):
         print(l)
         print('---')