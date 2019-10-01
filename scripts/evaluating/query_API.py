import urllib.parse
import urllib.request

url = 'https://www.uniprot.org/uploadlists/'

def getUniprotACC(pdb_ids):
   params = {
   'from': 'PDB_ID', #ACC+ID
   'to': 'ACC',
   'format': 'list',
   'query': ' '.join(pdb_ids)
   }

   data = urllib.parse.urlencode(params)
   data = data.encode('utf-8')
   req = urllib.request.Request(url, data)

   with urllib.request.urlopen(req) as f:
      response = f.read().decode('utf-8')
      uniprot_accs=response.splitlines()
      return uniprot_accs

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