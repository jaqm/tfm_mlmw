#!/usr/bin/env python3

import virustotal3
import os
from os.path import join, exists
import time
import json
import virustotal3.core

import config
from config import VT_API_KEY as API_KEY


class Virustotal_Client():

    def __init__(self, mw_samples_dir,
                 vt_waitting_time=config.vt_waitting_time):

        self._waitting_time = vt_waitting_time
        self.mw_samples_dir = mw_samples_dir

    def submit_file_virustotal(self, mw_filepath):

        vt = virustotal3.core.Files(API_KEY)

        response = vt.upload(mw_filepath)
        analysis_id = response['data']['id']
        print('Analysis ID: {}'.format(analysis_id))
        results = None

        print('Waiting for results ' + str(self._waitting_time) + ' secs ...')
        time.sleep(self._waitting_time)
        results = virustotal3.core.get_analysis(API_KEY, analysis_id)
        status = results['data']['attributes']['status']
        while 'completed' not in status:
            print('Current status: {}'.format(status))
            self._waitting_time += 3
            time.sleep(self._waitting_time/7)
            results = virustotal3.core.get_analysis(API_KEY, analysis_id)
            status = results['data']['attributes']['status']

        self._waitting_time -= 3
        return results

    def send_filenames(self, mw_filenames):

        mw_sent_list = []
        for mw_filename in mw_filenames:
            report_filepath = join(config.vt_reports_dir, mw_filename + "_vtreport.json")
            if not exists(report_filepath):
                mw_filepath = join(self.mw_samples_dir, mw_filename)
                vt_report = self.submit_file_virustotal(mw_filepath)
                print('Almacenando en disco duro..')
                with open(report_filepath, 'w') as outfile:
                    json.dump(vt_report, outfile, indent=4)
                mw_sent_list.append(mw_filename)

        return mw_sent_list

    def load_vt_report(self,
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
