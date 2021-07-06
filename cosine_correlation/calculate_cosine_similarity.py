#from scipy import stats
import numpy as np
import pandas as pd
import os
import math
import time
import datetime

cpu_count = min(16, os.cpu_count()) #os.cpu_count()
path_to_psite_changes = 'changed_phosphopeptides_sample.txt'
list_timepoint = ['1M', '3M', '6M']

def init(_d, _list_timepoint):
    global DF, TIME_POINT
    DF = _d
    TIME_POINT = _list_timepoint

def psite2vec(psite):
    v = np.zeros(len(TIME_POINT))
    x = DF.loc[DF.Psite_ID==psite,:]
    for i in range(x.shape[0]):
        idx_tp = TIME_POINT.index(x.time_point[i])
        v[idx_tp] = x.mean_ratio[i]
    return psite, v

def similarity_cos(x,y):
    if all([a==0 for a in x]) or all([b==0 for b in y]):
        return np.nan
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return np.dot(x,y) / (norm_x * norm_y)

def vector_similarity(site1, site2):
    v1 = psite2vec(site1)[1]
    v2 = psite2vec(site2)[1]
    return site1, site2, similarity_cos(v1, v2)
def sim_wrapper(args):
    return vector_similarity(*args)

if __name__ == '__main__':
    print("[multidimentional vector-based analysis]")
    print("number of cores = %d" % cpu_count)
    start_time = time.time()
    print(" > started at %s" % datetime.datetime.fromtimestamp(start_time))
    
    psite_changes = pd.read_table(path_to_psite_changes, sep='\t', index_col=0)
    p_sites = sorted(list(set(psite_changes.Psite_ID)))

    print(" > calculate cos_similarity")
    from multiprocessing import Pool
    pool = Pool(cpu_count, initializer=init, initargs=(psite_changes, list_timepoint))
    
    import itertools
    vector_similarity = pool.map(sim_wrapper, itertools.combinations(p_sites, 2))

    similarity = pd.DataFrame(vector_similarity, columns=['site1', 'site2', 'cos_sim'])
    possible_edges = similarity.loc[~np.isnan(similarity.cos_sim),:]
    candidate_edges = possible_edges.loc[np.abs(possible_edges.cos_sim)>0.9,:]
    similarity.to_csv("cos_similarity_all_pairs.tsv", sep='\t')
    possible_edges.to_csv("cos_similarity_possible_edges.tsv", sep='\t')
    candidate_edges.to_csv("cos_similarity_candidates.tsv", sep='\t')

    end_time = time.time()
    print(" > ended at %s" % datetime.datetime.fromtimestamp(end_time))
    elapsed_time = end_time - start_time
    print(" >> elapsed time = %s" % datetime.timedelta(seconds=elapsed_time))
