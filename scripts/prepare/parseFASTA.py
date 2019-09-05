'''
This script parses the uniref_100.fast file to extract 
the amino acid sequences.

For each protein in the dataset, the amino acid sequence 
in string format is extracted and saved on disk.

This file creates and save protein sequences like:
sequences=[seq1, seq2]
'''
from Bio import SeqIO
import pickle
import yaml

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

rawFilename='uniref_100.fasta'
sequences=[]

for idx, record in enumerate(SeqIO.parse(hyperparams['data_dir']+rawFilename, "fasta")):
    aa_sequence=str(record.seq)

    sequences.append(aa_sequence)

print('Number of sequences extracted:', len(sequences))

with open("sequences", "wb") as fp:
    pickle.dump(sequences, fp)