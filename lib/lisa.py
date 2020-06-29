#!/usr/bin/env python3

import requests
import json
from os.path import isfile, join, exists
from os import listdir

import config


def api_query(command, type_of_query="GET", task_id=None, url=config.base_url, params={'limit': config.dataset_size_max}, data=None, files=None):
    ''' Send any command to the LiSa api

    Params:
    - type_of_query: "GET" or "POST".
    - command: any of the values in config.get_queries or config.post_queries dicts.

    params: GET requests parameters
    data: POST requests parameters
    files: files to include in POST requests

    '''

    if type_of_query == "GET":
        query = config.get_queries
    elif type_of_query == "POST":
        query = config.post_queries

    api_query = query[command]

    if "<task_id>" in api_query:
        api_query = api_query.replace("<task_id>", task_id)

    if type_of_query == "GET":
        response = requests.get(url + api_query, params=params)
    elif type_of_query == "POST":
        response = requests.post(url + api_query, data=data, files=files)

    return response


def join_task_info(view, report, pcap, machinelog, output):
    ''' join all the information in the same json object.
    '''
    report['view'] = view
    report['pcap'] = pcap
    report['machinelog'] = machinelog
    report['output'] = output

    return report


def get_finished_tasks(params={'limit': config.dataset_size_max}):
    ''' Returns the finished tasks from the api
    '''
    tasks_info_finished = api_query("finished", params=params).json()

    if isinstance(tasks_info_finished, dict):
        if 'error' in tasks_info_finished.keys():
            # ipdb.set_trace()
            print(tasks_info_finished)
            print('ERROR AL OBTENER LAS TAREAS FINALIZADAS DEL API. Repetir petición?')
            exit()
    return tasks_info_finished


# Hard disk functions
def get_files_from_dir(mypath):
    ''' returns the files contained in a directory
    '''
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()
    return onlyfiles


def report_filepath(
        task_info,
        report_dir):

    if 'file_name' in task_info.keys():
        filename = task_info['file_name']
    elif 'filename' in task_info['result'].keys():
        filename = task_info['result']['filename']
    else:
        ipdb.set_trace()

    filepath = report_dir + "/" + filename + ".json"

    return filepath


def get_task_report_from_api(task_id):
    # informacion Task ID
    view = api_query('task_id_view', task_id=task_id).json()
    report = api_query('task_id_report', task_id=task_id).json()
    pcap = api_query('task_id_pcap', task_id=task_id).text
    machinelog = api_query('task_id_machinelog', task_id=task_id).text
    output = api_query('task_id_output', task_id=task_id).text

    task_report = join_task_info(view, report, pcap, machinelog, output)

    return task_report


def get_task_info_report(task_info):
    ''' Returns the task report stored in the hard disk or in the api (joined in a single json).
    @aparams:
    - task: task info taken directly from the /api/tasks/finished
    '''

    task_report = load_task_report(task_info)
    filename = task_info['result']['filename']
    if not task_report:
        print('Info: El reporte no se pudo cargar desde disco duro: ' +
              filename + ". Solicitando a API...")

        task_report = get_task_report_from_api(task_info['task_id'])

    return task_report


# LOAD/STORE FUNCTIONALITY
def store_task_report(
        task_report, overwrite=False,
        report_dir=config.reports_unknown_dir):
    ''' Saves the task_info to a file in the hard disk.
    '''

    if task_report['view']['status'] == 'SUCCESS':
        report_dir = config.reports_success_dir
    else:
        ipdb.set_trace()
        if task_report['view'] == '':
            report_dir = config.reports_failed_dir

    filepath = report_filepath(task_report, report_dir=report_dir)

    if overwrite or not exists(filepath):
        with open(filepath, 'w') as outfile:
            json.dump(task_report, outfile, indent=4)

    else:
        print('El fichero ya existe y/o no se sobreescribe: ' + filepath)


def load_task_report(
        task_info,
        report_dir=config.reports_success_dir):
    ''' If file report exist in the hard disk, returns its content, 
    e.o.c. None. 
    '''
    task_report = None

    filepath = report_filepath(task_info, report_dir=report_dir)
    if exists(filepath):
        with open(filepath, 'r') as infile:
            task_report = json.load(infile)

    return task_report


def submit_file(filename, exec_time="60", pretty="true"):

    command = "submit_full_analysis"
    filepath = join(config.malware_arm_dir, filename)
    files = {'file': open(filepath, 'rb')}
    data = {
        "exec_time": exec_time,
        "pretty": pretty
    }

    r = api_query(command, type_of_query="POST", files=files, data=data)
    return r


def send_files_to_sandbox(lof):
    ''' Send the list of files to the sandbox to run a full analysis.
    '''
    count = 0
    for filename in lof:
        submit_file(filename)
        count += 1
    print("Enviados a analizar: " + str(count))


def show_json_structure(json, sublevel=0):
    ''' Show the keys of a json object
    '''

    for key in json.keys():
        print(sublevel * " ", end='')
        print(key)
        if isinstance(json[key], dict):
            show_json_structure(json[key], sublevel+4)
