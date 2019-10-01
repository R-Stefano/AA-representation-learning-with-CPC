'''
This script is used to convert the ProteinNet dataset into csv files 
'''
import yaml
import numpy as np 
import pandas as pd
with open("../../hyperparams.yml", 'r') as f:
    configs = yaml.safe_load(f)

data_dir=configs['data_dir']
folder='raw/casp12'

files=['validation', 'testing', 'training_90']

for f_string in files:
    next_line_id=False
    next_line_primary_structure=False

    pdb_codes=[]
    aa_sequences=[]
    with open(data_dir+folder+'/'+f_string) as f:
        print('Processing file {}'.format(folder+'/'+f_string))
        for line in f:
            if next_line_id:
                idstring=line
                if '_' in idstring:
                    end=idstring.index('_')
                    id_parsed=idstring[end-4:end]
                elif '#' in idstring:
                    start=idstring.index('#')+1
                    id_parsed=idstring[start:start+4]
                else:
                    raise 'PDB ID not valid!'
                pdb_codes.append(id_parsed)

                next_line_id=False
            elif next_line_primary_structure:
                aa_sequences.append(line.rstrip())
                next_line_primary_structure=False

            if '[ID]' in line:
                next_line_id=True
            elif '[PRIMARY]' in line:
                next_line_primary_structure=True

    print(len(pdb_codes))
    print(len(aa_sequences))

    dataset={
        'pdb_id':pdb_codes,
        'seqs': aa_sequences
    }

    pd.DataFrame(dataset).to_csv(data_dir+'raw/csv/'+f_string+'.csv', index=False)