import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import mp
from scipy.stats import mannwhitneyu
from multiprocessing import Pool

def _strict_overlap(a, b):
    if a[1]<=b[1]<a[0]+a[1] or a[1]<b[0]+b[1]<=a[0]+a[1]:
        return True
    elif a[1]>=b[1] and a[0]+a[1]<=b[0]+b[1]:
        return True
    else:
        return False

def _loose_overlap(a, b):
    l = min(a[0], b[0])/2
    if a[1]<=b[1]<a[0]+a[1]-l or a[1]+l<b[0]+b[1]<=a[0]+a[1]:
        return True
    elif a[1]>=b[1] and a[0]+a[1]<=b[0]+b[1]:
        return True
    else:
        return False
    
def _remove_overlap(classes, candidates, cand_idx, overlap):
    cand_keep, idx_keep = {},{}
    for i in classes:
        i_cand, i_idx = [],[]
        if overlap == 'loose':
            print('loose')
            for cand, idx in zip(candidates[i],cand_idx[i]):
                if any([_loose_overlap(shapelet, idx) for shapelet in i_idx]):
                    pass
                else:
                    i_cand.append(cand), i_idx.append(idx)
        else:
            print('strict')
            for cand, idx in zip(candidates[i],cand_idx[i]):
                if any([_strict_overlap(selected, idx) for selected in i_idx]):
                    pass
                else:
                    i_cand.append(cand), i_idx.append(idx)
        cand_keep[i] = np.array(i_cand, dtype=object)
        idx_keep[i] = np.array(i_idx)
    return cand_keep, idx_keep

def _ddp_class(label, X, y, n_candidates, max_l):
    XA = np.concatenate([np.append(Ti, np.nan) for Ti in X[y==label]], axis=None)
    A_candidates = []
    A_cand_idx = []
    A_cand_quality = []
    for l in tqdm(range(3,min(200,max_l+1))):
        A,B = X[y==label], X[y!=label]
        
        #initialize distance distribution matrix
        pAA, pAB = [],[]
        
        # update ddm
        for instance in A:
            newValue=mp.stomp(np.nan_to_num(XA), l, instance)[0]
            pAA.append(newValue)
        pAA = np.array(pAA).transpose()
            
        for instance in B:
            newValue=mp.stomp(np.nan_to_num(XA), l, instance)[0]
            pAB.append(newValue)
        pAB = np.array(pAB).transpose()
        
        #quality measure
        quality = np.array([mannwhitneyu(sample1, sample2)[1] for sample1, sample2 in zip(pAA, pAB)])
        idx = np.argsort(quality)
        cand = np.array([XA[i:i+l] for i in idx])
        
        # pruning artificial candidates (concatenated tail and head of instances)
        mask = np.array([np.all(i==False) for i in np.isnan(cand)])
        for i in idx[mask][:n_candidates]:
            A_candidates.append(XA[i:i+l])
            A_cand_idx.append((l,i))
            A_cand_quality.append(quality[i])
            
    # sort candidates from same class by quality
    A_candidates = np.array(A_candidates, dtype=object)[np.argsort(A_cand_quality)]
    A_cand_idx = np.array(A_cand_idx)[np.argsort(A_cand_quality)]
    
    return (label, A_candidates, A_cand_idx)

def DDP_candidates(X, y, n_candidates=20, max_length=0.5, overlap=False):
    """candidates searching with quality evaluation by comparing
    distances distribution between target class instances and other classes
    
    Arguments
    ---------
        
    """
    classes = np.unique(y)
    max_l = int(len(min(X, key=len))*max_length)
    candidates={}
    cand_idx={}
    
    # concatenate target class instances and calculate MP against instances within same class and other classes 
    p = Pool(processes=len(classes))
    results = [p.apply_async(_ddp_class, args=(label, X, y, n_candidates, max_l)) for label in classes]
    p.close
    p.join
    for item in results:
        candidates[item.get()[0]]=item.get()[1]
        cand_idx[item.get()[0]]=item.get()[2]
        
    if not overlap:
        print('no')
        return candidates, cand_idx
    else:
        print('yes')
        cand_keep, idx_keep = _remove_overlap(classes, candidates, cand_idx, overlap=overlap)
        return cand_keep, idx_keep

def best_shapelets(classes, sorted_candidates, sorted_cand_idx, k=2):
    best_shapelets, best_s_idx = [],{}
    for label in classes:
        best_shapelets+=[sorted_candidates[label][i] for i in range(k)]
        best_s_idx[label]=sorted_cand_idx[label][:k]        
        
    return best_shapelets, best_s_idx


def transform(X, best_shapelets):
    transformed_X = np.array([[np.nan_to_num(mp.mass(ts, query), nan=np.inf).min() for ts in X] for query in best_shapelets]).real
    
    return transformed_X



