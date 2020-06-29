#!/usr/bin/env python3

from collections import defaultdict
import collections
import pickle
import ast
from json import load

from config import verbose, forbidden_chars, amount_of_words_to_keep, target_tag


def mergeListOfDicts(master_dict, sub_dict,
                     matching_key='file_name', sub_key=None):
    '''Merge two list of dicts using their common key.
    Params:
    - l1, l2: the list of dicts that will be merged.
    - matching_key: the common key.
    '''

    for e2 in sub_dict:
        for e1 in master_dict:
            if e1[matching_key] == e2[matching_key]:
                if sub_key:
                    e1[sub_key] = e2
                else:
                    e1.update(e2)
                break


def str2dict(input_data):

    input_data = input_data.replace('} {', '}, {')
    input_data = ast.literal_eval(input_data)
    result = []
    for value in input_data:
        if isinstance(value, dict):
            result.append(list(value.keys()))

    return result


def get_vector_from_value(input_data=[], check_strings=False):
    '''
    Returns the vector of word frequencies of the text received.
    '''

    if isinstance(input_data, str):
        if check_strings:
            if any([c in input_data for c in forbidden_chars]):
                bk()
                input_data = str2dict(input_data)

    elif isinstance(input_data, list):
        input_data = " ".join(input_data)
    else:
        if not (isinstance(input_data, bool) or
                isinstance(input_data, int) or
                input_data is None):
            bk()
        input_data = str(input_data)

    vector = collections.Counter(input_data.split(' '))

    return vector


def save_obj(obj, outfile):
    if verbose:
        print('Saving file: ' + outfile)
    with open(outfile, 'wb') as fp:
        pickle.dump(obj, fp)


def load_obj(infile):
    if verbose:
        print('Loading file: ' + infile)
    with open(infile, 'rb') as fp:
        try:
            itemlist = pickle.load(fp)
        except:
            with open(infile, 'r') as fp:

                itemlist = load(fp)

    return itemlist


def check_tfidf(dword, list_of_best_words):
    return len(dword) <= amount_of_words_to_keep and \
        all(word in list_of_best_words for word in dword)


def get_filtered_dataset(reports_db, filtered_dword_universe):
    '''Recibe un diccionario de palabras y devuelve el diccionario de
       ficheros finarios con las palabras que aparecen en filtered_dword_universe.
       Es la salida ideal para despues hacer un Dataframe.
    '''

    intersect = {}
    for report in reports_db:
        filename = report['file_name']
        existing_keys = \
            report.keys() & filtered_dword_universe.keys()

        intersect[filename] = {
            'dword': {word: report[word] for word in existing_keys},
            target_tag: report[target_tag].split('-')[0]

        }

    return intersect


def build_dfs(dword_universe, reports_db):

    if verbose:
        print('Removing words following the tfidf criteria.')
    result = {}
    amount_of_samples = len(reports_db)
    result[kind_of_report] = {}
    for k2, dword in report_info.items():

        words_tfidf = get_words_tfidf(
            dword, kind_of_report, k2, reports_db, amount_of_samples)

        list_of_best_words = get_best_words(
            words_tfidf, reports_db, kind_of_report, k2, dword_universe,
            amount_of_words_to_keep)

        if check_tfidf(dword_universe[kind_of_report][k2], list_of_best_words):
            if verbose:
                print("Congrats! The words you wanted are those which stay in the dataset. K1: "
                      + kind_of_report + " k2: " + k2)
            else:
                print('ERROR: Words are not being removed properly.')

        result[kind_of_report][k2] = \
            get_filtered_dataset(reports_db, dword_universe, kind_of_report, k2)

    return result
