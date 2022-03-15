# ************
# File: csv_reader.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Loads networks and specifications from a CSV file.
# ************

import os
import csv
import re

class QueryReader:

    def __init__(self):
        pass
   
    def read_from_csv(self, filepath):
        """
        Loads the verification queries from a csv file.

        Arguments:

            filepath: csv file of queries.

        Returns:

            A list of pairs of (network file, specification file)
        """
        queries= []
        basename = os.path.split(filepath)[0]

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                queries.append((
                    os.path.join(basename, row[0]), 
                    os.path.join(basename, row[1])
                ))

        return queries

    def read_from_file(self, nn_filepath, spec_filepath):
        """
        Loads the verification queries from folders/files.

        Arguments:

            nn_filepath: network file or folder of networks.
            
            spec_filepath: specification file or folder of specifications.

        Returns:

            A list of pairs of (network file, specification file)
        """
        queries = []
        nn_basename = os.path.split(nn_filepath)[0]
        spec_basename = os.path.split(nn_filepath)[0]

        # networks
        if os.path.isdir(nn_filepath):
            # network filenames
            nn_files = os.listdir(nn_filepath)
            # sort filenames by their numeric content
            nn_files = sorted(nn_files, key=lambda i: int(re.sub('[^0-9]', '', i)))
            nn_files = [os.path.join(nn_filepath, fl) for fl in nn_files]
        else:
            nn_files = [nn_filepath]

        # specifications
        if os.path.isdir(spec_filepath):
            # specification filenames
            spec_files = os.listdir(spec_filepath)
            # sort filenames by their numeric content
            spec_files = sorted(spec_files, key=lambda i: int(re.sub('[^0-9]', '', i)))
            spec_files = [os.path.join(spec_filepath, fl) for fl in spec_files]
        else:
            spec_files = [spec_filepath]

        return [ (i,j) for i in nn_files for j in spec_files ]






