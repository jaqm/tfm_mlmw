#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import average_precision_score, f1_score, recall_score, accuracy_score, precision_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import target_tag, num_cpus, num_splits, test_size, default_num_splits, index_column_name, image_dir, datamodels_list, default_type_of_average, types_of_average, chosen_average, chosen_metric, default_scoring
from lib.utils import timing


def get_class_counts(df, group_by_target=target_tag):
    grp = df.reset_index().groupby(group_by_target).nunique()[index_column_name]
    result = {key: grp[key] for key in list(grp.keys())}
    return result


def get_class_proportions(df):
    class_counts = get_class_counts(df)
    result = {val[0]: round(val[1]/df.shape[0], 4) for val in class_counts.items()}
    return result


def show_stats(df):

    class_counts = get_class_counts(df)
    class_proportions = get_class_proportions(df)

    print('Num classes:', len(class_counts))
    print('Class counts', class_counts)
    print('Class proportions:', class_proportions)


def apply_filter_on_df(df_source, df_filter, as_copy=True):
    '''
    df_source: the dataframe where the filter will be applied
    df_filter: the dataframe which represents the filter that will be applied on df_source.
    '''
    result = df_source[df_source.index.isin(df_filter.index)]
    if as_copy:
        result = result.copy()
    return result


def split_y(y, stratify=True, test_size=test_size):
    '''
    From a Dataframe, returns 2 groups of disjointed and stratified data.
    Usually used as train and test dataset.
    Ref: https://towardsdatascience.com/3-things-you-need-to-know-before-you-train-test-split-869dfabb7e50
    '''

    stratify_by = y if stratify else None

    train, test = train_test_split(
        y, test_size=test_size, stratify=stratify_by)

    return train, test


def _check_same_members_df_Xy(df):
    '''Checks X and y are non empty and contain the same members.
    '''
    df_filtered = apply_filter_on_df(df['X'], df['y'])
    same_amount_of_members = len(df['X']) == len(df['y']) == len(
        apply_filter_on_df(df['X'], df['y']))
    return same_amount_of_members and not df_filtered.empty


def evaluate_datamodel_pred(test_y, y_pred):  # , y_score):
    ''' Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    average:
    - 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
    - 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

    '''

    predicted_labels = np.unique(y_pred)

    result = {}
    for average in types_of_average:
        result[average] = {
            'precision': precision_score(test_y, y_pred, average=average, labels=predicted_labels),
            'accuracy': accuracy_score(test_y, y_pred),
            'recall': recall_score(test_y, y_pred, average=average),
            'f1': f1_score(test_y, y_pred, average=average),
        }

    return result


def cross_validate_wrap(datamodel, train_X, train_y):

    cv = StratifiedShuffleSplit(n_splits=num_splits, test_size=1/num_splits)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='weighted')
    }

    cross_val_score_var = cross_validate(
        datamodel, train_X, train_y, cv=cv, n_jobs=num_cpus, scoring=scoring)

    return cross_val_score_var


def printable_average_name(average):
    if average == None:
        average = 'Ninguna'
    elif average == 'binary':
        average = 'Binaria'
    elif average == 'samples':
        average = 'Por muestras'
    elif average == 'weighted':
        average = 'Ponderada'
    else:
        average = average.capitalize()

    return average


def confusion_matrix_wrapper(
        X_test, y_test, y_pred,
        algorithm_name, image_dir, target_names,
        classifier, sampler_name):

    disp = plot_confusion_matrix(
        classifier,
        X_test, y_test,
        display_labels=target_names,
        cmap=plt.cm.Blues,
        normalize='all')

    all_sample_title = algorithm_name
    plt.title(all_sample_title, size=15)

    image_filename = (sampler_name.replace(' ', '_')+'-'+algorithm_name).replace(" ", "_").lower()
    image_filepath = image_dir + "/" + image_filename + ".png"

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    plt.savefig(image_filepath)
    plt.close()

    return image_filename


def show_datamodel_stats(
        X_test, y_test, y_pred, datamodel_name,
        metrics, sampler_name,
        image_dir, target_names, image_filename,
        scoring=default_scoring):

    precision = metrics['precision']
    accuracy = metrics['accuracy']
    recall = metrics['recall']
    f1 = metrics['f1']

    figname = image_filename.replace(".png", "")
    fig_path = "graphics/TFM_2020/confusion_matrix/"+figname

    print("\\begin{minipage}{0.6\\textwidth}")
    print("\centering")
    print("\includegraphics[width=1\\linewidth]{"+fig_path+"}")
    print("\end{minipage}")
    print("\hfill")
    print('\\begin{minipage}{0.5\\textwidth}\\raggedright')
    print("Metrica:", printable_average_name(chosen_average))
    print("\\begin{itemize}")
    print("\item F1: ", "{:.3f}".format(f1))
    print("\item Accuracy: ", "{:.3f}".format(accuracy))
    print("\item Recall: " + "{:.3f}".format(recall))
    print("\item Precision: " + "{:.3f}".format(precision))
    print("\end{itemize}")
    print("\end{minipage}")
    print()


def printable_sampler_name(sampler_name):
    if sampler_name == 'Random_Sampler':
        sampler_name = 'Aleatorio'
    elif sampler_name == 'No Sampler':
        sampler_name = 'Ninguno'

    return sampler_name


def print_sampling_stats(sampler_name, result, test_y, le, X_test):
    '''
    @params:
    ev: is the dict which contains the evaluation scores for a datamodel
    '''

    target_names = le.classes_.tolist()
    print('\subsection{Muestreo:', printable_sampler_name(sampler_name), '}')
    for datamodel_count, datamodel_name in enumerate(result[sampler_name]):

        pred_y = result[sampler_name][datamodel_name]['pred_y']
        print('\\subsubsection{Modelo:', datamodel_name, '}')

        image_filename = confusion_matrix_wrapper(
            X_test, test_y,
            result[sampler_name][datamodel_name]['pred_y'],
            datamodel_name, image_dir, target_names,
            result[sampler_name][datamodel_name]['trained_datamodel'],
            sampler_name)

        show_datamodel_stats(
            X_test, le.inverse_transform(test_y),
            pred_y,
            datamodel_name,
            result[sampler_name][datamodel_name]['evaluation'][chosen_average],
            sampler_name,
            image_dir,
            target_names, image_filename
        )


def train_algorithms(df, sampler, le,
                     image_dir=image_dir,
                     scoring=default_scoring):
    '''
    @params:
    df: data containing train and test, X and y info.
    sampler: the sampler that will be used before the datamodel

    scoring: ref: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    '''

    result = {}

    for (datamodel_name, clf) in datamodels_list:
        classifier = make_pipeline(sampler, clf)

        classifier.fit(
            df['train']['X'],
            df['train']['y'].values.ravel())
        pred_y = classifier.predict(df['test']['X'])

        result[datamodel_name] = {
            'sampler_name': list(classifier.named_steps.keys())[0],
            'trained_datamodel': classifier,
            'pred_y': le.inverse_transform(pred_y),
            'evaluation': evaluate_datamodel_pred(df['test']['y'],
                                                  pred_y)
        }

    return result


def get_latex_table(result, average):

    print('\\subsection{Tabla de resultados - media', printable_average_name(average), '}')
    print('\\begin{table}[H]')
    print('\centering')
    print('\label{tabla_comparativa_' + average+'}')
    print('\\resizebox{\\textwidth}{!}{')
    print('\\begin{tabular}{|l|l|c|c|c|c|c|}')
    print("\\hline")
    print("\\textbf{Sampleado} & \\textbf{Algoritmo} & \\textbf{F1} & \\textbf{Exhaustividad} & \\textbf{Precisión} &  \\textbf{Exactitud}\\\\")
    print("\\hline")

    for sampler_name in result:
        line_counter = 0
        for datamodel_name, data in result[sampler_name].items():
            if round(len(result[sampler_name])/2) == line_counter:
                print(printable_sampler_name(sampler_name), end='')
            line_counter += 1

            print(' & ',
                  datamodel_name, ' & ',
                  "{:.3f}".format(data['evaluation'][average]['f1']), '  & ',
                  "{:.3f}".format(data['evaluation'][average]['recall']), '  & ',
                  "{:.3f}".format(data['evaluation'][average]['precision']), ' & ',
                  "{:.3f}".format(data['evaluation'][average]['accuracy']),
                  '\\\\')
        print("\\hline")

    print('\\end{tabular}}')
    print('\caption{Tabla de resultados - Media:', printable_average_name(average), '}')
    print('\\end{table}')


def get_best_datamodel(result):
    '''
    Check the results of the stage data and returns the data model with higest scorer.

    The default scorer is F1 because it takes into account recall and precision, both important for our purpose: to tag malware.

    @parameters:
    - result: the result of the analysis

    @returns:
    - best_datamodel_name: the name of the chosen datamodel.
    - best_datamodel: the (previously trained) sklearn datamodel.
    '''

    bd_name = None
    bd = None
    bd_score = 0

    for sampler_name in result:
        for datamodel_name, data in result[sampler_name].items():
            score = data['evaluation'][chosen_average][chosen_metric]
            if score > bd_score:
                best_sampler = sampler_name
                bd_name = datamodel_name
                bd = result[best_sampler][bd_name]['trained_datamodel']
                bd_score = score

    return best_sampler, bd_name, bd, bd_score


def label_encode(y):
    le = LabelEncoder()
    le.fit(y[target_tag].astype(str))
    y[target_tag] = le.transform(y[target_tag].astype(str))

    return y, le


def remove_items_with_less_members_than(df, min_members):
    '''Remove members in df with less that min_members will be removed.
    '''

    classes, df_indices = np.unique(df, return_inverse=True)

    class_counts = np.bincount(df_indices)
    positions = np.where(class_counts < min_members)[0].tolist()

    classes_to_remove = [classes[position]
                         for position in positions]

    df = df[~df[target_tag].isin(classes_to_remove)]

    return df


def remove_duplicated_columns(df):
    '''
     Removes duplicated colums
    '''
    print('Removing duplicated columns..')
    old_cols = len(df.columns)
    df = df.transpose().drop_duplicates().transpose()
    print('Duplicated columns removed:',
          old_cols - len(df.columns))
    print('Unique columns remained:', len(df.columns))

    return df


def clean_dword_universe(dword_universe, features):
    ''' Return a dword universe containing only 
    the keys in the features variable.
    '''
    result = {}
    for word in features:
        result[word] = dword_universe[word]
    return result
