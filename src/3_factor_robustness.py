import os
import numpy as np
import pandas as pd

from processing.robustness import subsample, run_method, reorder_factors, calculate_abs_corr, robustness

# load data without outcome measures
df = pd.read_csv("../data/behavior.csv", index_col=0).iloc[:, 6:]

if not os.path.exists("../outputs/robustness"):
    os.makedirs("../outputs/robustness")

robustness(df, 6, "fa", "../outputs/robustness", rotation="promax")