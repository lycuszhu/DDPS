import pandas as pd
import numpy as np
import os
import mp
from sklearn.model_selection import train_test_split

def load_dataset(folder, file, random_state, split=False):
    # load original UCR train test
    train = pd.read_csv(os.path.join(folder,file,file+'_TRAIN.tsv'),sep="\t",header=None,on_bad_lines='skip')
    test = pd.read_csv(os.path.join(folder,file,file+'_TEST.tsv'),sep="\t",header=None,on_bad_lines='skip')
    train_size, test_size = len(train), len(test)
    
    # resampling
    df = pd.concat([train,test],ignore_index=True)
    y = df.pop(0)
    X = df.to_numpy()
    classes = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    if not split:
        return X_train, X_test, y_train, y_test, classes    
    
    else:
        X_spl, X_clf, y_spl, y_clf = train_test_split(X_train, y_train, test_size = 0.5)
        return X_spl, X_clf, X_test, y_spl, y_clf, y_test, classes

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
    
def remove_overlap(classes, candidates, cand_idx, overlap):
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

def keep_shapelets(classes, sorted_candidates, sorted_cand_idx, k=2):
    keep_shapelets, keep_s_idx = [],{}
    for label in classes:
        keep_shapelets+=[sorted_candidates[label][i] for i in range(k)]
        keep_s_idx[label]=sorted_cand_idx[label][:k]        
        
    return keep_shapelets, keep_s_idx


def transform(X, best_shapelets):
    transformed_X = np.array([[mp.mass(np.nan_to_num(ts), query).real.min() for ts in X] for query in best_shapelets])
    
    return transformed_X