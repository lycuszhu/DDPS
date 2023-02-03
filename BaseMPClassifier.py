import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
#import mass_ts as mts
import mp
from scipy.signal import find_peaks
from multiprocessing import Pool

def _baseline_class(label,X,y,n_candidates,max_l):
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

def baseline_candidates(X, y, n_candidates=20, max_length=0.5):
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
    
    # concatenate target class and rest classes into long series    
    p = Pool(processes=len(classes))
    results = [p.apply_async(_baseline_class, args=(label, X, y, n_candidates, max_l)) for label in classes]
    p.close
    p.join
    for item in results:
        candidates[item.get()[0]]=item.get()[1]
        cand_idx[item.get()[0]]=item.get()[2]
    
    return candidates, cand_idx

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
    
def sort_by_quality(X, y, n_candidates=20, max_length=0.5, overlap=False):
    """sort candidates by quality measures
    quality measures: run two-sample t-test of candidate distances between target class instances and other classes instances, using the p-value as the quality measure.
    
    Arguments
    ---------
    
    """
    classes = np.unique(y)
    candidates, cand_idx = baseline_candidates(X, y, n_candidates, max_length)
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
        cand_keep, idx_keep = _remove_overlap(classes, sorted_candidates, sorted_cand_idx, overlap=overlap)
        return cand_keep, idx_keep


def best_shapelets(classes, sorted_candidates, sorted_cand_idx, k=2):
    best_shapelets, best_s_idx = [],{}
    for label in classes:
        best_shapelets+=[sorted_candidates[label][i] for i in range(k)]
        best_s_idx[label]=sorted_cand_idx[label][:k]        
        
    return best_shapelets, best_s_idx


def transform(X, best_shapelets):
    transformed_X = np.array([[np.nan_to_num(mp.mass(ts, query),nan=np.inf).min() for ts in X] for query in best_shapelets]).real
    
    return transformed_X