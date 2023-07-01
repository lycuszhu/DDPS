import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import mp
import utils
from scipy.signal import find_peaks
from multiprocessing import Pool

def _prototypeMP_class(label,X,y,n_candidates,max_l):
    XA = np.concatenate([np.append(Ti, np.nan) for Ti in X[y==label]], axis=None)
    XB = np.concatenate([np.append(Ti, np.nan) for Ti in X[y!=label]], axis=None)
    
    # calculating |PAA-PAB| as candidate indicators
    A_candidates=[]
    A_cand_idx=[]
    
    for l in tqdm(range(3,min(200,max_l+1))):
        pAA, indexAA = mp.stomp(np.nan_to_num(XA),l)
        pAB, indexAB = mp.stomp(np.nan_to_num(XA),l,np.nan_to_num(XB))
        d = np.abs(pAB-pAA)
        
        # find top candidates with find_peaks
        peaks,_ = find_peaks(d)
        if len(peaks)!=0:
            k = min(n_candidates, len(peaks))            
            # sort peaks index by heights in descending order
            idx = peaks[np.argsort(-d[peaks])]
            
        else:
            k = n_candidates
            idx = np.argsort(-d)
        
        # pruning artificial candidates (concatenated tail and head of instances)
        cand = np.array([XA[i:i+l] for i in idx])
        mask = np.array([np.all(i==False) for i in np.isnan(cand)])
        for i in idx[mask][:k]:
            A_candidates.append(XA[i:i+l])
            A_cand_idx.append((l,i))
            
    # return candidates and index
    return (label, A_candidates, A_cand_idx)

def prototypeMP_candidates(X, y, n_candidates=20, max_length=0.5, overlap=False):
    """candidates searching with matrix profile
    collect top n candidates of each length with
    scipy.signal.find_peak package
    
    Arguments
    ---------
    X : array of shape [n_samples, n_timepoints], capable of processing unequal length series
    
    y : array of shape [n_samples]
    
    n_candidates : int, maximum number of candidates to keep of each length
    
    max_length : float between 0 and 1, maximum size of candidate in proportion of longest instance length; size cap currently set at 100
    
    """
    classes = np.unique(y)
    max_l = int(len(min(X, key=len))*max_length)
    candidates={}
    cand_idx={}    
    
    # parallel running candidate searching from each class on multiple cores
    p = Pool(processes=len(classes))
    results = [p.apply_async(_prototypeMP_class, args=(label, X, y, n_candidates, max_l)) for label in classes]
    p.close
    p.join
    for item in results:
        candidates[item.get()[0]]=item.get()[1]
        cand_idx[item.get()[0]]=item.get()[2]
    
    # sort candidates by quality measure
    sorted_candidates={}
    sorted_cand_idx={}
    for label in classes:
        distances=np.array([[mp.mass(ts, query).min() for ts in X] for query in candidates[label]]).real
        quality=[]
        for d in distances:
            target_class = d[y==label].tolist()
            rest_class = d[y!=label].tolist()
            quality.append(stats.ttest_ind(target_class,rest_class)[1])
        sorted_candidates[label]=np.array(candidates[label],dtype=object)[np.argsort(quality)]
        sorted_cand_idx[label]=np.array(cand_idx[label])[np.argsort(quality)]
    
    if not overlap:
        print('no')
        return sorted_candidates, sorted_cand_idx
    else:
        print('yes')
        cand_keep, idx_keep = utils.remove_overlap(classes, sorted_candidates, sorted_cand_idx, overlap=overlap)
        return cand_keep, idx_keep