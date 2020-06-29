#!/usr/bin/env python3


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from os import path
from PIL import Image
import sys
import traceback
import logging
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from deco import concurrent, synchronized
from imblearn.over_sampling import RandomOverSampler

from lib.loader import save_obj, load_obj
from config import target_tag, train_X_filepath, le_filepath, df_stage1_filepath, num_cpus, num_splits, test_size, default_num_splits, stage2_train_X_df_filepath, stage2_test_X_df_filepath, master_y_encoded_filepath, dword_universe_flatten_filepath, test_master_db_filepath, dword_universe_train_X_filepath, avclass_train_y_filepath, avclass_test_y_filepath, df_splitted_filepath, apply_oversample, samplers_list, datamodels_list
from lib.dataframe_handler import get_class_counts, get_class_proportions, show_stats, apply_filter_on_df, evaluate_datamodel_pred, train_algorithms, get_best_datamodel, get_latex_table, print_sampling_stats, types_of_average


def remove_keys(test_master_db, keys_to_remove):
    result = []
    for report in test_master_db:
        for key in keys_to_remove:
            report.pop(key, None)


def main():

    df_splitted = load_obj(df_splitted_filepath)
    le = load_obj(le_filepath)

    result = {}

    for (sampler_name, sampler) in samplers_list:
        result[sampler_name] = train_algorithms(
            df_splitted, sampler, le)
        print_sampling_stats(
            sampler_name, result,
            df_splitted['test']['y'].values.ravel(),
            le, df_splitted['test']['X'])

    for average in types_of_average:
        get_latex_table(result, average)

    best_sampler_name, best_datamodel_name, best_datamodel, f1_score = get_best_datamodel(result)

    print()
    print('Mejor algoritmo según por criterio F1 ponderado para sampler',
          best_sampler_name, ': ', best_datamodel_name)

    return


if __name__ == "__main__":
    sys.exit(main())
