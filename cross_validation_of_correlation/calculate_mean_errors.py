import numpy as np
import pandas as pd
import math
import os

GroupName = '5xFAD'
Dir_diff = 'cv_result'
Dataset1 = 'val12'
Dataset2 = 'val3'
Dir_output = '.'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", help="group name", type=str)
parser.add_argument("--set1", help="dataset name for estimation", type=str)
parser.add_argument("--set2", help="dataset name for validation", type=str)
parser.add_argument("--indir", help='directory for input (difference values of every edge)', type=str)
parser.add_argument("--outdir", help='directory for storing results', type=str)
args = parser.parse_args()

if args.g is not None:
    GroupName = args.g
if args.set1 is not None:
    Dataset1 = args.set1
if args.set2 is not None:
    Dataset2 = args.set2
if args.indir is not None:
    Dir_diff = args.dir_diff
if args.outdir is not None:
    Dir_output = args.outdir

path_input = os.path.join(
    Dir_diff,
    "Diff_similarity_%s-%s-%s_possible_edges_with_all_values.tsv" % (GroupName, Dataset1, Dataset2)
)

df = pd.read_table(path_input, sep='\t', index_col='ID')
# exclude incomplete data
df = df.loc[df.diff_sq>=0.0,:]
n = df.shape[0]
# mean absolute error
mae = np.sum(np.abs(df.loc[:, "diff"])) / n
sd = np.std(np.abs(df.loc[:, "diff"]))
# mean squared error
mse = np.sum(df.diff_sq) / n
sq_sd = np.std(df.diff_sq)
# root mean square error
rmse = mse**(1/2)

# output to console
print('\t'.join([str(x) for x in [GroupName, Dataset1, Dataset2, n, mae, sd, mse, rmse, sq_sd]]))
