#!/usr/bin/env python3

''' This tool's purpose is to interact with the LiSa API and the malware dataset in order to provide an
 interface able to send files to LiSa sandbox a retrieve it's results.
'''

import argparse

from lib.lisa import *
from lisa_launcher.lisa_launcher import *
import config
from lib.virustotal import *


def get_stats():

    tasks_failed = api_query("failed").json()
    print("Tareas fallidas: " + str(len(tasks_failed)))
    tasks_finished = api_query("finished").json()
    print("Tareas correctas: " + str(len(tasks_finished)))
    tasks_pending = api_query("pending").json()
    print("Tareas pendientes: " + str(len(tasks_pending)))


def _get_analyzed_files_in_db(finished_tasks=None, sort_them=True):

    if not finished_tasks:
        finished_tasks = get_finished_tasks()

    filenames_analyzed = [
        task_info['result']['filename']
        for task_info in finished_tasks
        if 'result' in task_info.keys() and
        'filename' in task_info['result'].keys() and
        task_info['status'] == "SUCCESS"
    ]

    if sort_them:
        filenames_analyzed.sort()

    return filenames_analyzed


def _get_task_infos_not_yet_stored():

    finished_tasks = get_finished_tasks()
    analyzed_files_in_db = _get_analyzed_files_in_db(finished_tasks=finished_tasks)
    print("Total de reportes en DB: " + str(len(analyzed_files_in_db)))
    files_mw_arm_reports = get_files_from_dir(config.reports_success_dir)
    print('Total de reportes en disco duro: ' + str(len(files_mw_arm_reports)))
    filenames_not_yet_stored = [f for f in analyzed_files_in_db if f +
                                ".json" not in files_mw_arm_reports]
    print("Total de reportes aún no almacenados en disco duro: " + str(len(filenames_not_yet_stored)))

    count = 0
    task_info = None

    task_info_not_yet_stored = [
        task_info for task_info in finished_tasks
        if task_info['status'] == "SUCCESS" and
        task_info['result']['filename'] in filenames_not_yet_stored
    ]

    print('Se han encontrado ' + str(len(task_info_not_yet_stored)) + ' para almacenar en disco duro.')

    return task_info_not_yet_stored


def update_reports_collection(mw_samples_dir=config.malware_arm_dir):

    files_mw_arm = get_files_from_dir(mw_samples_dir)
    print('Muestras de malware arm en total: ' + str(len(files_mw_arm)))

    task_infos_not_yet_stored = _get_task_infos_not_yet_stored()

    if task_infos_not_yet_stored:
        count = 0
        print('Almacenando en disco duro..')
        for task_info in task_infos_not_yet_stored:
            count += 1
            task_report = get_task_info_report(task_info)
            store_task_report(task_report)
        print("Se almacenaron " + str(count) + " reportes en disco duro.")

    return None


def send_pending_to_sandbox(mw_samples_dir=config.malware_arm_dir):

    files_mw_arm = get_files_from_dir(mw_samples_dir)
    print('Muestras de malware arm en total: ' + str(len(files_mw_arm)))

    filenames_analyzed = _get_analyzed_files_in_db()
    samples_analysis_pending = [
        filename for filename in files_mw_arm
        if filename not in filenames_analyzed
    ]
    print('Ficheros pendientes de analizar en base de datos: ' + str(len(samples_analysis_pending)))

    task_infos_not_yet_stored = _get_task_infos_not_yet_stored()
    if len(task_infos_not_yet_stored) == 0:
        send_files_to_sandbox(samples_analysis_pending)
    else:
        print("Aún hay reportes sin almacenar en base de datos. Store them and run this again.")

    return None


def send_to_virustotal(mw_samples_dir):

    mw_filenames = get_files_from_dir(mw_samples_dir)
    vtc = Virustotal_Client(mw_samples_dir)

    filenames_sent = vtc.send_filenames(mw_filenames)

    print("Se almacenaron " + str(len(filenames_sent)) + " reportes virustotal en disco duro.")
    return filenames_sent


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tool to manage samples and LiSa sandbox.')
    parser.add_argument('--samples_dir', help='malware samples directory',
                        default=config.malware_arm_dir, required=False)
    parser.add_argument('--update_reports_collection',
                        help='update the reports collection taking the information from the LiSa API.',
                        action="store_true", required=False)
    parser.add_argument('--virustotal', help='send the files to virustotal',
                        action="store_true", required=False)
    parser.add_argument('--send_pending_to_sandbox',
                        help='Send the report pending samples to LiSa sandbox.',
                        action="store_true", required=False)
    parser.add_argument('--pending', help='show the pending analysis in LiSa.',
                        action="store_true", required=False)
    parser.add_argument('--console',
                        help='Open a python console',
                        action="store_true", required=False)
    args = parser.parse_args()

    if args.update_reports_collection:
        update_reports_collection(config.malware_arm_dir)

    elif args.send_pending_to_sandbox:
        send_pending_to_sandbox(args.samples_dir)
    elif args.virustotal:
        send_to_virustotal(args.samples_dir)
    elif args.pending:
        print(api_query('pending').json())

    if args.console:
        investigation()
        ipdb.set_trace()
        print("Welcome to ipdb console.")
