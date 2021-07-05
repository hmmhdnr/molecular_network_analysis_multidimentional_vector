import pandas as pd
from itertools import combinations, groupby
from multiprocessing import Pool
import time
import datetime
import os
import random

bootstrap_N = 1000000
Dir_output = 'result'
cpu_count = min(16, os.cpu_count())
TaskID = 0

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-B", help="number of bootstrap samples", type=int)
parser.add_argument("--task_id", help="Task ID for parallel", type=int)
args = parser.parse_args()

if args.B is not None:
    bootstrap_N = args.B
if args.task_id is not None:
    TaskID = args.task_id

def split_iter(iterable, N=3):
    for i, item in groupby(enumerate(iterable), lambda x: x[0] // N):
        yield (x[1] for x in item)

def init(_comb):
    global COMBINED, LIST_LEN
    COMBINED = _comb
    LIST_LEN = len(_comb)

def overwrap_ratio(lst_idx):
    lst = [COMBINED[x] for x in lst_idx]
    rest = [COMBINED[x] for x in range(LIST_LEN) if x not in lst_idx]
    intersect = set(rest) & set(lst)
    return len(lst), len(rest), len(intersect), len(intersect)/len(COMBINED), ','.join(intersect)

def overwrap_ratio_wrapper(args):
    return overwrap_ratio(*args)

if __name__ == '__main__':
    path_ad = "ad_core_nodes.csv"
    path_ftld = "ftld_core_nodes.csv"
    path_common = "common_core_nodes.csv"

    ad = pd.read_table(path_ad, sep=',', usecols=['FullName', 'GeneSymbol', 'shared name'])
    ftld = pd.read_table(path_ftld, sep=',', usecols=['FullName', 'GeneSymbol', 'shared name'])
    common = pd.read_table(path_common, sep=',', usecols=['FullName', 'GeneSymbol', 'shared name'])

    ad_proteins = list(ad.loc[:,'shared name'])
    ftld_proteins = list(ftld.loc[:,'shared name'])
    common_proteins = list(common.loc[:,'shared name'])
    print("#ad = %d, #ftld = %d, #common = %d" % (len(ad_proteins), len(ftld_proteins), len(common_proteins)))

    combined_list = ad_proteins + ftld_proteins
    x = ','.join(combined_list)
    print("#combined list = %d" % len(combined_list))

    pool = Pool(cpu_count, initializer=init, initargs=(combined_list,))
    result = pool.map(overwrap_ratio, [random.sample(range(len(combined_list)), k=len(ftld_proteins)) for x in range(bootstrap_N)])

    print("#results = %d" % len(result))
    x = pd.DataFrame(result, columns=['FTLD', 'AD', 'Common', 'overwrap_ratio', 'list_intersect'])
    x.to_csv(os.path.join(Dir_output, "permutation_result_core_network_task%d.tsv" % TaskID), sep='\t')

    ratios = x.loc[:,"overwrap_ratio"]
    expected_ratio = len(common_proteins) / len(combined_list)
    r = [o for o in ratios if o>expected_ratio]
    print("probability of expected ratio (%f) is %f for %d permutations" % (expected_ratio, len(r)/len(ratios), len(ratios)))
    
    pool.close()
    pool.terminate()
