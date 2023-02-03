#!/usr/bin/env python
# coding: utf-8


#### Mueen's Algorithm for Similarity Selection (no numeric error) ####
import numpy as np
from mass_ts.core import moving_average, moving_std
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
from random import random

def zEuclidean(x,y):
    return euclidean(np.nan_to_num(zscore(x)), np.nan_to_num(zscore(y)))
    
def mass(ts, query):
    """
    Compute the distance profile for the given query over the given time 
    series. Optionally, the correlation coefficient can be returned.
    Parameters
    ----------
    ts : array_like
        The array to create a rolling window on.
    query : array_like
        The query.
    Returns
    -------
    An array of distances.
    Raises
    ------
    ValueError
        If ts is not a list or np.array.
        If query is not a list or np.array.
        If ts or query is not one dimensional.
    """
    #ts, query = mtscore.precheck_series_and_query(ts, query)

    n = len(ts)
    m = len(query)
    x = ts
    y = query
    e = np.full(m,0)
    r = [random() for i in range(m)]

    meany = np.mean(y)
    sigmay = np.std(y)
    
    meanx = moving_average(x, m)
    meanx = np.append(np.ones([1, len(x) - len(meanx)]), meanx)
    sigmax = moving_std(x, m)
    sigmax = np.append(np.zeros([1, len(x) - len(sigmax)]), sigmax)
    w = np.argwhere(sigmax==0)[m-1:] - (m-1) # find where sigmax is 0
    
    y = np.append(np.flip(y), np.zeros([1, n - m]))
    
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Y.resize(X.shape)
    Z = X * Y
    z = np.fft.ifft(Z)
    
    # avoid numeric errors
    if not sigmay==0 and not np.any(sigmax==0):
        dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / 
                    (sigmax[m - 1:n] * sigmay))
        dist = np.sqrt(dist.astype(complex))
        
    elif not sigmay==0 and np.any(sigmax==0):
        #print('x has constant')
        dist = 2 * (m - (z[m - 1:n] - m * meanx[m - 1:n] * meany) / 
                    (sigmax[m - 1:n] * sigmay))
        dist = np.sqrt(dist.astype(complex))
        dist[w] = zEuclidean(e, query)
    
    else:
        #print('y has constant')
        dist = np.full(n-m+1,zEuclidean(r, query))
        dist[w] = 0
        
    return dist

#### Matrix Profile STOMP algorithm (fit for tsA!=tsB, no numeric errors) ####

import numpy.fft as fft

def is_array_like(a):
    """
    Helper function to determine if a value is array like.
    Parameters
    ----------
    a : obj
        Object to test.
    Returns
    -------
    True or false respectively.
    """
    return isinstance(a, tuple([list, tuple, np.ndarray]))

def to_np_array(a):
    """
    Helper function to convert tuple or list to np.ndarray.
    Parameters
    ----------
    a : Tuple, list or np.ndarray
        The object to transform.
    Returns
    -------
    The np.ndarray.
    Raises
    ------
    ValueError
        If a is not a valid type.
    """
    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)

def _clean_nan_inf(ts):
    """
    Converts tuples & lists to Numpy arrays and replaces nan and inf values with zeros
    Parameters
    ----------
    ts: Time series to clean
    """

    #Convert time series to a Numpy array
    ts = to_np_array(ts)

    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    return ts


def _self_join_or_not_preprocess(tsA, tsB, m):
    """
    Core method for determining if a self join is occuring and returns appropriate
    profile and index numpy arrays with correct dimensions as all np.nan values.
    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    tsB: Time series to compare the query against. Note that, if no value is provided, ts_b = ts_a by default.
    m: Length of subsequence to compare.
    """
    n = len(tsA)
    #if tsB is not None: # matrix profile length should always be len(tsA)-m+1
        #n = len(tsB)

    shape = n - m + 1

    return (np.full(shape, np.inf), np.full(shape, np.inf))

def is_self_join(tsA, tsB):
    """
    Helper function to determine if a self join is occurring or not. When tsA 
    is absolutely equal to tsB, a self join is occurring.
    Parameters
    ----------
    tsA: Primary time series.
    tsB: Subquery time series.
    """
    return tsB is None or np.array_equal(tsA, tsB)

def movmeanstd(ts,m):
    """
    Calculate the mean and standard deviation within a moving window passing across a time series.
    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] -sSq[:-m]

    movmean = segSum/m
    a = segSumSq / m - (segSum/m) ** 2
    a[np.argwhere(a<0)]=0
    movstd = np.sqrt(a)

    return [movmean,movstd]

def slidingDotProduct(query,ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. 
    Note that we use Numpy's rfft method instead of fft.
    
    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """

    m = len(query)
    n = len(ts)


    #If length is odd, zero-pad time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]


    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, 
    #which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))
    
    #Note that we only care about the dot product results from index m-1 onwards, 
    #as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]

def DotProductStomp(prev_query,query,ts,m,dot_first,dot_prev,order):
    """
    Updates the sliding dot product for a time series ts from the previous dot product dot_prev.
    Parameters
    ----------
    ts: Time series under analysis.
    m: Length of query within sliding dot product.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    order: The location of the first point in the query.
    """

    l = len(ts)-m+1
    dot = np.roll(dot_prev,1)

    dot += query[m-1]*ts[m-1:l+m]-prev_query[0]*np.roll(ts[:l],1) # problem here

    #Update the first value in the dot product array
    dot[0] = dot_first[order]

    return dot

def massStomp(prev_query,query,ts,dot_first,dot_prev,index,mean,std):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries using the STOMP dot product speedup. Note that we are returning the square of MASS.
    Parameters
    ----------
    query: Time series snippet to evaluate. Note that, for modified STOMP, the query does not must be a subset of ts.
    ts: Time series to compare against query.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    index: The location of the first point in the query.
    mean: Array containing the mean of every subsequence in ts.
    std: Array containing the standard deviation of every subsequence in ts.
    """
    m = len(query)
    meanq = np.mean(query)
    sigmaq = np.std(query)
    e = np.full(m,0)
    r = [random() for i in range(m)]
    dot = DotProductStomp(prev_query,query,ts,m,dot_first,dot_prev,index)

    #Return both the MASS calcuation and the dot product
    #res = 2*m*(1-(dot-m*mean[index]*mean)/(m*std[index]*std))
    
    # avoid numeric errors
    if not sigmaq==0 and not np.any(std==0):
        res = 2 * (m - (dot - m * mean * meanq) / 
                    (std * sigmaq))
        res = np.sqrt(res.astype(complex))
            
    elif not sigmaq==0 and np.any(std==0):
        res = 2 * (m - (dot - m * mean * meanq) / 
                    (std * sigmaq))
        res = np.sqrt(res.astype(complex))
        res[np.argwhere(std==0)] = zEuclidean(e, query)
    
    else:
        res = np.full(len(std),zEuclidean(r, query))
        res[np.argwhere(std==0)] = 0

    return res, dot

def STOMPDistanceProfile(tsA,idx,m,tsB,dot_first,dp,mean,std):
    """
    Returns the distance profile of a query within tsA against the time series tsB using the even more efficient iterative STOMP calculation. Note that the method requires a pre-calculated 'initial' sliding dot product.
    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, for the time being, only tsB = tsA is allowed
    dot_first: The 'initial' sliding dot product, or QT(1,1) in Zhu et.al
    dp: The dot product between tsA and the query starting at index m-1
    mean: Array containing the mean of every subsequence of length m in tsA (moving window)
    std: Array containing the mean of every subsequence of length m in tsA (moving window)
    """

    selfJoin = is_self_join(tsA, tsB)
    if selfJoin:
        tsB = tsA

    query = tsB[idx:(idx+m)]
    n = len(tsA)

    #Calculate the first distance profile via MASS
    if idx == 0:
        distanceProfile = mass(tsA, query).real        

        #Currently re-calculating the dot product separately as opposed to updating all of the mass function...
        dot = slidingDotProduct(query,tsA)

    #Calculate all subsequent distance profiles using the STOMP dot product shortcut
    else:
        prev_query = tsB[idx-1:(idx+m-1)]
        res, dot = massStomp(prev_query,query,tsA,dot_first,dp,idx,mean,std)
        distanceProfile = res.real


    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))
        distanceProfile[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    #Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distanceProfile,np.full(n-m+1,idx,dtype=float)), dot

from matrixprofile import order

def _matrixProfile_stomp(tsA,m,orderClass,distanceProfileFunction,tsB=None):    
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)    

    if not is_array_like(tsB):
        tsB = tsA
    order = orderClass(len(tsB)-m+1)

    tsA = _clean_nan_inf(tsA)
    tsB = _clean_nan_inf(tsB)

    idx=order.next()

    #Get moving mean and standard deviation
    mean, std = movmeanstd(tsA,m)

    #Initialize code to set dot_prev to None for the first pass
    dp = None

    #Initialize dot_first to None for the first pass
    dot_first = slidingDotProduct(tsA[:m],tsB)

    while idx != None:

        #Need to pass in the previous sliding dot product for subsequent distance profile calculations
        (distanceProfile,querySegmentsID),dot_prev = distanceProfileFunction(tsA,idx,m,tsB,dot_first,dp,mean,std)

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

        dp = dot_prev
    return (mp,mpIndex)


def stomp(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed according to the directed STOMP procedure.
    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile_stomp(tsA,m,order.linearOrder,STOMPDistanceProfile,tsB)

