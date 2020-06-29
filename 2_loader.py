#!/usr/bin/env python3

import argparse
from os import listdir
from os.path import isfile, join, exists
import json
from flatten_dict import flatten
from collections import Counter
import pandas as pd
import concurrent
from deco import concurrent, synchronized

from config import lisa_db_filepath, vt_db_filepath, reports_success_dir, vt_reports_dir, clamscan_report_filepath, master_db_filepath,  report_keys_w_raw_data, keys_wo_data, min_samples_per_class, verbose, unknown_malware_tag, target_tag, fields_to_ignore, forbidden_chars, num_cpus, dword_universe_filepath, essential_keys, master_y_filepath, original_master_db_filepath, clamav_y_filepath, avclass_y_filepath, avclass_train_y_filepath, avclass_test_y_filepath, le_filepath, master_y_encoded_filepath, train_master_db_filepath, test_master_db_filepath, amount_of_words_to_keep
from lib.loader import mergeListOfDicts, get_vector_from_value, save_obj, load_obj, build_dfs
from lib.dataframe_handler import split_y, label_encode, remove_items_with_less_members_than, show_stats
from lib.utils import timing
from lib.dataset_utils import get_dword_universe, remove_words_under_X_appearances

# NOTE:
# - task_info is the information in the format it comes from the api from the /api/task/<finished|pending|failure> endpoints.
# - task_report, is the information in the unified format. Containing all the information stored in the database.
#


def load_reports(reports_dir):
    ''' Returns the existing reports
    '''
    reports_filepaths = [
        join(reports_dir, f)
        for f in listdir(reports_dir) if isfile(join(reports_dir, f))
    ]

    reports = []
    for filename in listdir(reports_dir):
        filepath = join(reports_dir, filename)
        report = load_obj(filepath)
        if 'file_name' not in report.keys():
            # Removes '_vtreport' from the end of the filename.
            report['file_name'] = filename[:-14]
        reports.append(report)

    return reports


def load_db_reports(db_filepath, reports_dir=None):

    bk()
    reports = []
    if isfile(db_filepath):
        if '.pkl' in db_filepath:
            reports = load_obj(db_filepath)
        else:
            with open(db_filepath, 'r') as db_file:
                reports = json.load(db_file)
    else:
        reports = load_reports(reports_dir)

        if not exists(db_filepath):
            save_obj(reports, db_filepath)

    return reports


def read_cs_reports(cs_report_filepath=clamscan_report_filepath):

    cs_reports = []
    with open(cs_report_filepath, 'r') as cs_file:
        for line in cs_file.readlines():
            if 'VirusShare' not in line:
                continue
            line = line.replace('\n', '')
            if '/' in line:
                value = line.split('/')[3]
            filename = line.split(':')[0]
            # We keep the type of malware, not it's specific name.
            clamscan_result = line.split(' ')[1].split('-')[0]
            result = {
                'file_name': filename,
                target_tag: clamscan_result
            }
            cs_reports.append(result)

    return cs_reports


def mark_unknown_malware(master_db):
    ''' Mark the untagged malware as 'UNKOWN'.
    '''
    for report in master_db:
        if target_tag not in report.keys():
            report[target_tag] = unknown_malware_tag


def remove_samples_with_less_appearances(
        master_db, min_samples_per_type):
    ''' Removes the samples that belong to a target_tag
    with few appearances.

    This is made because we need at least 1 sample to train and test.
    '''

    cs_tags = {item['file_name']: item[target_tag]
               for item in master_db}
    counter = Counter(cs_tags.values())
    filtered_names = [key for key, value in counter.items()
                      if value >= min_samples_per_type]

    filtered_master_db = [report for report in master_db
                          if report[target_tag] in filtered_names]

    return filtered_master_db


def normalized_value(report_value):
    result = None

    if not report_value:
        result = 0
    elif isinstance(report_value, str):
        result = Counter(report_value.split())
    elif isinstance(report_value, int):
        result = report_value
    elif isinstance(report_value, dict):
        bk()
        result = Counter(report_value.values)
    elif isinstance(report_value, bool):
        result = Counter(str(report_value).split())
    elif isinstance(report_value, list):
        if not report_value[0]:
            result = 0
        elif isinstance(report_value[0], dict):
            result = [' '.join(map(str, x.values())) for x in report_value]

            result = ' '.join(map(str, result))
            result = Counter(result.split())
        else:
            result = ' '.join(map(str, report_value))
            result = Counter(result.split())

    return result


@timing
def normalize_dict_as_bag_of_words(master_db):
    '''
    The goal of this function is to return a bag of words dataframe.

    parameters:
    -----------

    master_db: dictionare containing any amount of
        columns, containing any amount of dicts, integer, text, etc.

    returns:
    --------

    master_db: dictionare containing the renamed features,
    and their repetition value while maintaining their information context.

    '''

    if verbose:
        print("Normalizing master_db..")
    master_db = [flatten(report, reducer='underscore')
                 for report in master_db]

    for pos, report in enumerate(master_db):
        print("\r\t> Progress\t:{:.2%}".format((pos)/len(master_db)), end='', flush=True)

        for key in report:
            if not any([value in key for value in essential_keys]):
                report[key] = normalized_value(report[key])

    print()
    master_db = [flatten(report, reducer='underscore')
                 for report in master_db]

    return master_db


def clean_dataset(master_db,
                  keys_to_remove=keys_wo_data):

    removed_keys = []
    for report in master_db:
        for context in report:
            for key in keys_to_remove:
                if context == key:
                    removed_keys.append(context)

    for key in removed_keys:
        for report in master_db:
            report.pop(key, None)

    for report in master_db:
        del report['static_analysis']['binary_info']['size']
        del report['static_analysis']['binary_info']['os']
        del report['static_analysis']['binary_info']['arch']

    return


def prepare_master_db(master_db, load_vt_reports=False):
    ''' Clean and flatten all the dataset. 
    Note: No feature selection is applied.
    '''

    if verbose:
        print("Preparing master_db..")
    clean_dataset(master_db)
    master_db = normalize_dict_as_bag_of_words(master_db)
    flatten_db(master_db)

    return master_db


def flatten_report(report, report_keys_w_raw_data=report_keys_w_raw_data):

    for key in report:
        if key in report_keys_w_raw_data:
            if isinstance(report[key], dict):
                report[key] = flatten(report[key],
                                      reducer='underscore')

    return report


def flatten_db(master_db):
    if verbose:
        print('Aplanando master_db..')
    for report in master_db:
        report = flatten_report(report)


def get_filename_for_md5(master_db, hash_md5):

    file_name = None
    for report in master_db:
        if report['md5'] == hash_md5:
            file_name = report['file_name']

    return file_name


def get_y_avclass(master_db_orig):
    fichero_AVClass = '../../Reports/AVCLASS.result'
    result = []
    counter = 0
    with open(fichero_AVClass, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            if len(line) > 1 and len(line) < 4:
                counter += 1
                print("\r\t> Progress\t:{:.2%}".format((counter)/len(master_db)), end='', flush=True)
                hash_md5 = line[0]
                result.append({
                    'file_name':  get_filename_for_md5(master_db, hash_md5),
                    'avclass_tag': line[1]
                })

    result = pd.DataFrame(result).set_index('file_name')

    return result


def split_master_db(master_db, train_y, test_y):
    train_y_filenames = list(train_y.index)
    test_y_filenames = list(test_y.index)
    train_master_db = []
    test_master_db = []
    for report in master_db:
        if report['file_name'] in train_y_filenames:
            train_master_db.append(report)
        elif report['file_name'] in test_y_filenames:
            test_master_db.append(report)

    return train_master_db, test_master_db


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tool to run the machine learning analysis.')

    args = parser.parse_args()

    master_db = load_db_reports(lisa_db_filepath, reports_success_dir)
    y = get_y_avclass(master_db)
    # Ensure there are at least min_samples_per_class
    # samples of each class in the dataset.
    # NOTE: least 2 elements of the same class are required
    # in order to train the algorithms.
    y = remove_items_with_less_members_than(y,
                                            min_samples_per_class)

    master_db = [report for report in master_db
                 if report['file_name'] in y.index]
    keys = [x['file_name'] for x in master_db]
    y = y.loc[keys]

    # Split to get the stats
    train_y, test_y = split_y(y)
    print("Stats for the whole dataset:")
    show_stats(y)
    print("Stats for train_y:")
    show_stats(train_y)
    print("Stats for test_y:")
    show_stats(test_y)

    # le: Label Enconder. Used to encode-decode strings labels
    # (like target_tag) as numbers.
    y, le = label_encode(y)
    # Real split of samples: training and test sets.
    train_y, test_y = split_y(y)

    master_db = prepare_master_db(master_db)

    train_master_db, test_master_db = split_master_db(master_db, train_y, test_y)

    dword_universe = get_dword_universe(train_master_db)

    ############################################
    # Train_X Filter: Minimum word appearances #
    ############################################
    remove_words_under_X_appearances(
        train_master_db, dword_universe)

    save_obj(le, le_filepath)
    save_obj(train_master_db, train_master_db_filepath)
    save_obj(test_master_db, test_master_db_filepath)
    save_obj(train_y, avclass_train_y_filepath)
    save_obj(test_y, avclass_test_y_filepath)
    save_obj(dword_universe, dword_universe_filepath)

    exit(0)
