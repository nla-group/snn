import time
import math
import numpy as np
from sklearn import datasets
from classix import loadData

from sklearn.metrics.cluster import normalized_mutual_info_score
from dcnn.native.dbscan import dbscan as native_dbscan
from dcnn.snn.dbscan import dbscan as snn_dbscan

def sigificant_digit(number, digits=4):
    return round(number, digits - int(math.floor(math.log10(abs(number)))) - 1)

 
xtrain, ytrain = loadData('Banknote')
xtrain = (xtrain - xtrain.mean(axis=0)) / xtrain.std(axis=0)
print("shape:", xtrain.shape)

eps = [0.1, 0.2, 0.3, 0.4, 0.5]
seeds = np.arange(5)

runtime_brute = np.zeros((len(eps)))
runtime_kt = np.zeros((len(eps)))
runtime_bt = np.zeros((len(eps)))
runtime_snn = np.zeros((len(eps)))

for random_seed in seeds:

    for j in range(len(eps)):
        bc = native_dbscan(eps=eps[j], algorithm='dbscan', query='brute')

        st = time.time()
        bc.fit(xtrain)
        et = time.time()
        runtime_brute[j] = et - st
        
        kt = native_dbscan(eps=eps[j], algorithm='dbscan', query='kd_tree')

        st = time.time()
        kt.fit(xtrain)
        et = time.time()
        runtime_kt[j] = et - st

        bt = native_dbscan(eps=eps[j], algorithm='dbscan', query='ball_tree')

        st = time.time()
        bt.fit(xtrain)
        et = time.time()
        runtime_bt[j] = et - st

        snn = snn_dbscan(eps=eps[j], algorithm='dbscan')
        st = time.time()
        snn.fit(xtrain)
        et = time.time()
        runtime_snn[j] = et - st
        
        if random_seed == seeds[0]:
            print("NMI: ", sigificant_digit(normalized_mutual_info_score(ytrain, bc.labels)))
        
        
        
with open('result/runtimeBFBanknote.npy', 'wb') as f:
    np.save(f, runtime_brute)

with open('result/runtimeKTBanknote.npy', 'wb') as f:
    np.save(f, runtime_kt)

with open('result/runtimeBTBanknote.npy', 'wb') as f:
    np.save(f, runtime_bt)

with open('result/runtimeSNNBanknote.npy', 'wb') as f:
    np.save(f, runtime_snn)
    
    
    
xtrain, ytrain = loadData('Dermatology')
nonans = np.isnan(xtrain).sum(1) == 0
xtrain = xtrain[nonans,:]
ytrain = ytrain[nonans]
xtrain = (xtrain - xtrain.mean(axis=0)) / xtrain.std(axis=0)
print("shape:", xtrain.shape)

eps = [5, 5.1, 5.2, 5.3, 5.4]
seeds = np.arange(5)

runtime_brute = np.zeros((len(eps)))
runtime_kt = np.zeros((len(eps)))
runtime_bt = np.zeros((len(eps)))
runtime_snn = np.zeros((len(eps)))

for random_seed in seeds:

    for j in range(len(eps)):
        bc = native_dbscan(eps=eps[j], algorithm='dbscan', minPts=15, query='brute')

        st = time.time()
        bc.fit(xtrain)
        et = time.time()
        runtime_brute[j] += et - st
        
        
        kt = native_dbscan(eps=eps[j], algorithm='dbscan', query='kd_tree')

        st = time.time()
        kt.fit(xtrain)
        et = time.time()
        runtime_kt[j] = et - st

        bt = native_dbscan(eps=eps[j], algorithm='dbscan', query='ball_tree')

        st = time.time()
        bt.fit(xtrain)
        et = time.time()
        runtime_bt[j] = et - st

        snn = snn_dbscan(eps=eps[j], algorithm='dbscan')
        st = time.time()
        snn.fit(xtrain)
        et = time.time()
        runtime_snn[j] = et - st
        
        
        if random_seed == seeds[0]:
            print("NMI: ", sigificant_digit(normalized_mutual_info_score(ytrain, bc.labels)))
            
        
with open('result/runtimeBFDermatology.npy', 'wb') as f:
    np.save(f, runtime_brute)

with open('result/runtimeKTDermatology.npy', 'wb') as f:
    np.save(f, runtime_kt)

with open('result/runtimeBTDermatology.npy', 'wb') as f:
    np.save(f, runtime_bt)

with open('result/runtimeSNNDermatology.npy', 'wb') as f:
    np.save(f, runtime_snn)
    
    
    
    
xtrain, ytrain = loadData('Ecoli')
xtrain = (xtrain - xtrain.mean(axis=0)) / xtrain.std(axis=0)

eps = [0.5, 0.6, 0.7, 0.8, 0.9]
seeds = np.arange(5)

runtime_brute = np.zeros((len(eps)))
runtime_kt = np.zeros((len(eps)))
runtime_bt = np.zeros((len(eps)))
runtime_snn = np.zeros((len(eps)))

for random_seed in seeds:

    for j in range(len(eps)):
        bc = native_dbscan(eps=eps[j], algorithm='dbscan', query='brute')

        st = time.time()
        bc.fit(xtrain)
        et = time.time()
        runtime_brute[j] = et - st
        
        
        kt = native_dbscan(eps=eps[j], algorithm='dbscan', query='kd_tree')

        st = time.time()
        kt.fit(xtrain)
        et = time.time()
        runtime_kt[j] = et - st

        bt = native_dbscan(eps=eps[j], algorithm='dbscan', query='ball_tree')

        st = time.time()
        bt.fit(xtrain)
        et = time.time()
        runtime_bt[j] = et - st

        snn = snn_dbscan(eps=eps[j], algorithm='dbscan')
        st = time.time()
        snn.fit(xtrain)
        et = time.time()
        runtime_snn[j] = et - st 
        
        
        if random_seed == seeds[0]:
            print("NMI: ", sigificant_digit(normalized_mutual_info_score(ytrain, bc.labels)))
       
        
with open('result/runtimeBFEcoli.npy', 'wb') as f:
    np.save(f, runtime_brute)

with open('result/runtimeKTEcoli.npy', 'wb') as f:
    np.save(f, runtime_kt)

with open('result/runtimeBTEcoli.npy', 'wb') as f:
    np.save(f, runtime_bt)

with open('result/runtimeSNNEcoli.npy', 'wb') as f:
    np.save(f, runtime_snn)
    
    
    
    
xtrain , ytrain = loadData('Phoneme')
nonans = np.isnan(xtrain).sum(1) == 0
xtrain = xtrain[nonans,:]
ytrain = ytrain[nonans]
xtrain = (xtrain - xtrain.mean(axis=0)) / xtrain.std(axis=0)

eps = [8.5, 8.6, 8.7, 8.8, 8.9]
seeds = np.arange(5)

runtime_brute = np.zeros((len(eps)))
runtime_kt = np.zeros((len(eps)))
runtime_bt = np.zeros((len(eps)))
runtime_snn = np.zeros((len(eps)))

for random_seed in seeds:

    for j in range(len(eps)):
        bc = native_dbscan(eps=eps[j], algorithm='dbscan', minPts=15, query='brute')

        st = time.time()
        bc.fit(xtrain)
        et = time.time()
        runtime_brute[j] = et - st
        
        
        kt = native_dbscan(eps=eps[j], algorithm='dbscan', query='kd_tree')

        st = time.time()
        kt.fit(xtrain)
        et = time.time()
        runtime_kt[j] = et - st

        bt = native_dbscan(eps=eps[j], algorithm='dbscan', query='ball_tree')

        st = time.time()
        bt.fit(xtrain)
        et = time.time()
        runtime_bt[j] = et - st

        snn = snn_dbscan(eps=eps[j], algorithm='dbscan')
        st = time.time()
        snn.fit(xtrain)
        et = time.time()
        runtime_snn[j] = et - st
        
        if random_seed == seeds[0]:
            print("NMI: ", sigificant_digit(normalized_mutual_info_score(ytrain, bc.labels)))
        
        
with open('result/runtimeBFPhoneme.npy', 'wb') as f:
    np.save(f, runtime_brute)

with open('result/runtimeKTPhoneme.npy', 'wb') as f:
    np.save(f, runtime_kt)

with open('result/runtimeBTPhoneme.npy', 'wb') as f:
    np.save(f, runtime_bt)

with open('result/runtimeSNNPhoneme.npy', 'wb') as f:
    np.save(f, runtime_snn)
    
    
    
    
xtrain, ytrain = loadData('Wine')
xtrain = (xtrain - xtrain.mean(axis=0)) / xtrain.std(axis=0)

eps = [2.2, 2.3, 2.4, 2.5, 2.6]
seeds = np.arange(5)

runtime_brute = np.zeros((len(eps)))
runtime_kt = np.zeros((len(eps)))
runtime_bt = np.zeros((len(eps)))
runtime_snn = np.zeros((len(eps)))

for random_seed in seeds:

    for j in range(len(eps)):
        bc = native_dbscan(eps=eps[j], algorithm='dbscan', query='brute')

        st = time.time()
        bc.fit(xtrain)
        et = time.time()
        runtime_brute[j] = et - st
        
        
        kt = native_dbscan(eps=eps[j], algorithm='dbscan', query='kd_tree')

        st = time.time()
        kt.fit(xtrain)
        et = time.time()
        runtime_kt[j] = et - st

        bt = native_dbscan(eps=eps[j], algorithm='dbscan', query='ball_tree')

        st = time.time()
        bt.fit(xtrain)
        et = time.time()
        runtime_bt[j] = et - st

        snn = snn_dbscan(eps=eps[j], algorithm='dbscan')
        st = time.time()
        snn.fit(xtrain)
        et = time.time()
        runtime_snn[j] = et - st
        
        if random_seed == seeds[0]:
            print("NMI: ", sigificant_digit(normalized_mutual_info_score(ytrain, bc.labels)))
        
with open('result/runtimeBFWine.npy', 'wb') as f:
    np.save(f, runtime_brute)

with open('result/runtimeKTWine.npy', 'wb') as f:
    np.save(f, runtime_kt)

with open('result/runtimeBTWine.npy', 'wb') as f:
    np.save(f, runtime_bt)

with open('result/runtimeSNNWine.npy', 'wb') as f:
    np.save(f, runtime_snn)
