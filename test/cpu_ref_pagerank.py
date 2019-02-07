#!/usr/bin/python

# Usage : python3 cpu_ref_pagerank.py graph.mtx alpha

#import numpy as np
import sys
import time
from scipy.io import mmread
import numpy as np
import networkx as nx
import os

# Command line arguments
argc = len(sys.argv)
if argc<=2:
    print("Error: usage is : python3 cpu_ref_pagerank.py graph.mtx alpha")
    sys.exit()
mmFile = sys.argv[1]
alpha = float(sys.argv[2])
print('Reading '+ str(mmFile) + '...')
#Read
M = mmread(mmFile).asfptype()

M = M.tocsr()
if M is None :
    raise TypeError('Could not read the input graph')
if M.shape[0] != M.shape[1]:
    raise TypeError('Shape is not square')

# should be autosorted, but check just to make sure
if not M.has_sorted_indices:
    print('sort_indices ... ')
    M.sort_indices()

z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

print (M.shape[0])
Gnx = nx.DiGraph(M)

print('Solving... ')
t1 = time.time()
pr = nx.pagerank(Gnx, alpha=alpha, nstart = z, max_iter=5000, tol = 1e-10)
t2 =  time.time() - t1
print('Time : '+str(t2))

print(pr)

b = open(os.path.splitext(os.path.basename(mmFile))[0] + '_pagerank.txt', "w")
for k,v in pr.items():
    b.write(str(k)+" " +str(v) + "\n")        
b.close()
print ("Wrote pagerank to the file: "+ os.path.splitext(os.path.basename(mmFile))[0] + '_pagerank.txt')

print('Done')
