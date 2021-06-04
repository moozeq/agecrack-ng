#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from anage import AnAgeDatabase
from ncbi import NCBIDatabase

data_dir = 'data'

agecrack_ng_description = """
                                                             888                              
                                                             888                              
                                                             888                              
 8888b.   .d88b.   .d88b.   .d8888b 888d888 8888b.   .d8888b 888  888       88888b.   .d88b.  
    "88b d88P"88b d8P  Y8b d88P"    888P"      "88b d88P"    888 .88P       888 "88b d88P"88b 
.d888888 888  888 88888888 888      888    .d888888 888      888888K 888888 888  888 888  888 
888  888 Y88b 888 Y8b.     Y88b.    888    888  888 Y88b.    888 "88b       888  888 Y88b 888 
"Y888888  "Y88888  "Y8888   "Y8888P 888    "Y888888  "Y8888P 888  888       888  888  "Y88888 
              888                                                                         888 
         Y8b d88P                                                                    Y8b d88P 
          "Y88P"                                                                      "Y88P"  
          
Tool for searching and extracting age-related features from data.
"""


def load_logger(verbosity: int):
    try:
        log_level = {
            0: logging.ERROR,
            1: logging.WARN,
            2: logging.INFO}[verbosity]
    except KeyError:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)


def create_dirs_structure():
    Path(f'{data_dir}/proteomes').mkdir(parents=True, exist_ok=True)


def load_anage_db(filename: str) -> AnAgeDatabase:
    return AnAgeDatabase(filename, data_dir)


def load_ncbi_db(filename: str) -> NCBIDatabase:
    return NCBIDatabase(filename, data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=agecrack_ng_description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--anage', type=str, default=f'{data_dir}/anage_data.txt', help='AnAge database file')
    parser.add_argument('--ncbi', type=str, default=f'{data_dir}/eukaryotes.txt', help='NCBI eukaryotes database file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity')
    args = parser.parse_args()

    load_logger(args.verbose)
    create_dirs_structure()
    anage_db = load_anage_db(args.anage)
    ncbi_db = load_ncbi_db(args.ncbi)
    anage_db.analyze(ncbi_db)
