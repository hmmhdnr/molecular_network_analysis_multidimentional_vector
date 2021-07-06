import pandas as pd
from itertools import groupby
from multiprocessing import Pool
import time
import datetime
import os
import random

# permutation-based simulation
# 2021/5/27 by H.Homma

ListSize = 1965
Size_A = 73
Size_B = 62
Expected_Overwrap_Size = 43
bootstrap_N = 10000
Dir_output = 'result_from_all_proteins'
cpu_count = min(16, os.cpu_count())
TaskID = 0

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task_id", help="Task ID for parallel", type=int)
parser.add_argument("--size_a", help="size of list_A", type=int)
parser.add_argument("--size_b", help="size of list_B", type=int)
parser.add_argument("--population_size", help="size of population", type=int)
parser.add_argument("-B", help="number of bootstrap trials", type=int)
parser.add_argument("-E", help="Expected overwrap size", type=int)
parser.add_argument("-d", help="output directory", type=str)
args = parser.parse_args()

if args.task_id is not None:
    TaskID = args.task_id
if args.size_a is not None:
    Size_A = args.size_a
if args.size_b is not None:
    Size_B = args.size_b
if args.population_size is not None:
    ListSize = args.population_size
if args.B is not None:
    bootstrap_N = args.B
if args.E is not None:
    Expected_Overwrap_Size = args.E
if args.d is not None:
    Dir_output = args.d
if not os.path.isdir(Dir_output):
    os.makedirs(Dir_output)

def init(_a, _b, _l):
    global A, B, ALL
    A = _a
    B = _b
    ALL = _l

def split_iter(iterable, N=3):
    for i, item in groupby(enumerate(iterable), lambda x: x[0] // N):
        yield (x[1] for x in item)

def overwrap_ratio(idx):
    la = random.sample(range(ALL), A)
    lb = random.sample(range(ALL), B)
    intersect = sorted(list(set(la) & set(lb)))
    union = sorted(list(set(la) | set(lb)))
    return idx, len(la), len(lb), len(intersect), len(union), len(intersect)/len(union), ','.join([str(x) for x in intersect]), ','.join([str(x) for x in union])

def overwrap_ratio_wrapper(args):
    return overwrap_ratio(*args)

if __name__ == '__main__':
    expected_list_size = Size_A + Size_B - Expected_Overwrap_Size
    expected_ratio = Expected_Overwrap_Size / expected_list_size
    print("bootstrap size_A = %d, size_B = %d from population of size %d" % (Size_A, Size_B, ListSize))
    print("expected list size = %d, expected overwrap size = %d, expected overwrap ratio = %f" % (expected_list_size, Expected_Overwrap_Size, expected_ratio))

    pool = Pool(cpu_count, initializer=init, initargs=(Size_A, Size_B, ListSize))
    result = pool.map(overwrap_ratio, range(bootstrap_N))

    x = pd.DataFrame(result, columns=['index', 'size_a', 'size_b', 'size_intersect', 'size_union', 'overwrap_ratio', 'intersection', 'union']).set_index('index')
    x.to_csv(os.path.join(Dir_output, "permutation_result_for_core_network_task%d.tsv" % TaskID), sep='\t')

    ratios = x.loc[:,"overwrap_ratio"]
    r = [o for o in ratios if o >= expected_ratio]
    print("%d samples were greater than or equal to the expected value" % len(r))
    print("probability of expected ratio (%f) is %f for %d bootstrap samples" % (expected_ratio, len(r)/len(ratios), len(ratios)))
    
    pool.close()
    pool.terminate()
