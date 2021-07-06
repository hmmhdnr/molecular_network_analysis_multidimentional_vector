import pandas as pd
import os
import time
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", help="a group name", type=str)
parser.add_argument("--set1", help="a name of validation dataset 1", type=str)
parser.add_argument("--set2", help="a name of validation dataset 2", type=str)
parser.add_argument("-d", help="directory path to similarity data", type=str)
args = parser.parse_args()

# example: 5xFAD, (sample1, sample2) vs (sample3)
GroupName = '5xFAD'
Dataset1 = 'val12'
Dataset2 = 'val3'
DataDir = 'cos_sim/possible_edges'

if args.g is not None:
    GroupName = args.g
if args.set1 is not None:
    Dataset1 = args.set1
if args.set2 is not None:
    Dataset2 = args.set2
if args.d is not None:
    DataDir = args.d

cpu_count = min(16, os.cpu_count()) #os.cpu_count()

def init(_d1, _d2):
    global DS1, DS2
    DS1 = _d1
    DS2 = _d2

def diff_sim(idx):
    x1 = DS1.similarity.loc[idx]
    x2 = DS2.similarity.loc[idx]
    diff = x2-x1
    if x1==0 or x2==0:
        diff_sq = -1
    else:
        diff_sq = diff**2
    return idx, diff, diff_sq

if __name__ == '__main__':
    print("[cross-validation of similarity of edges]")
    path_output = os.path.join(".", "Diff_similarity_%s-%s-%s_possible_edges_with_all_values.tsv" % (GroupName, Dataset1, Dataset2))
    path_set1 = os.path.join(DataDir, "%s_%s_cos_similarity_possible_edges_with_all_values.tsv" % (GroupName, Dataset1))
    path_set2 = os.path.join(DataDir, "%s_%s_cos_similarity_possible_edges_with_all_values.tsv" % (GroupName, Dataset2))
    ds1 = pd.read_table(path_set1, sep='\t')
    ds2 = pd.read_table(path_set2, sep='\t')
    id_ds1 = ds1.site1.str.cat(ds1.site2, sep='_')
    id_ds2 = ds2.site1.str.cat(ds2.site2, sep='_')
    unique_ids = sorted(list(set(id_ds1) & set(id_ds2)))
    ds1 = pd.DataFrame({'Edge_ID': id_ds1, 'similarity': ds1.cos_sim}).set_index('Edge_ID')
    ds2 = pd.DataFrame({'Edge_ID': id_ds2, 'similarity': ds2.cos_sim}).set_index('Edge_ID')
    result = pd.DataFrame(index=unique_ids, columns=['node1', 'node2', 'sim1', 'sim2', 'diff', 'diff_sq'])

    print(" > calculate diff of similarity values using multiprocessing.Pool")
    from multiprocessing import Pool
    pool = Pool(cpu_count, initializer=init, initargs=(ds1, ds2))
    diff = pool.map(diff_sim, unique_ids)
    diff = pd.DataFrame(diff, columns=['ID', 'diff', 'diff_sq']).set_index('ID')
    diff = diff.loc[diff.diff_sq>0,:]
    diff.to_csv(path_output, sep='\t')
    print("difference between %s and %s in %s was calculated successfuly." % (Dataset1, Dataset2, GroupName))
