import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import mp
import utils
from multiprocessing import Pool

def _update(existingAggregate, newValue, count):
    """Welford's online algorithm to update mean and variance while new value added to distribution"""
    (mean,M2,variance)=existingAggregate
    delta=newValue-mean
    mean+=delta/count
    delta2=newValue-mean
    M2+=delta*delta2
    variance=M2/count
    return (mean,M2,variance)

def _qualityMeasure(pAA,pAB,nA,nB):
    """p-value of two-sample t-test as quality measure
    """
    (meanA,M2A,varianceA)=pAA
    (meanB,M2B,varianceB)=pAB
    tt=(meanA-meanB)/np.sqrt(varianceA/nA+varianceB/nB)
    pval=stats.t.sf(np.abs(tt),nA+nB-2)*2 #two-sided pvalue
    return pval

def _ddp_class(label, X, y, n_candidates, max_l):
    XA = np.concatenate([np.append(Ti, np.nan) for Ti in X[y==label]], axis=None)
    A_candidates = []
    A_cand_idx = []
    A_cand_quality = []
    for l in tqdm(range(3,min(200,max_l+1))):
        A,B = X[y==label], X[y!=label]
        nA, nB = len(A), len(B)
        
        # initialize distance distribution info
        pAA = np.zeros((3,len(XA)-l+1))
        pAB = np.zeros((3,len(XA)-l+1))
        
        # update mean, M2, variance after each MP calculation
        for i in range(nA):
            newValue=mp.stomp(np.nan_to_num(XA), l, A[i])[0]
            pAA = _update(pAA, newValue, i+1)
            
        for i in range(nB):
            newValue=mp.stomp(np.nan_to_num(XA), l, B[i])[0]
            pAB = _update(pAB, newValue, i+1)
            
        # quality measure
        quality=_qualityMeasure(pAA,pAB,nA,nB)
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
    
    return (label,A_candidates,A_cand_idx)

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
    
    # parallel running candidate searching from each class on multiple cores 
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
        cand_keep, idx_keep = utils.remove_overlap(classes, candidates, cand_idx, overlap=overlap)
        return cand_keep, idx_keep
