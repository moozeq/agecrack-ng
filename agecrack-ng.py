#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

from anage import AnAgeDatabase
from mmseq import run_mmseqs_pipeline
from ncbi import NCBIDatabase
from prot_encode import ESM
from random_forest import RF
from uniprot import download_proteomes_by_names
from utils import extract_proteins, save_records, count_records, load_and_add_results

dir_data = 'data'
dir_results = 'results'

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
    Path(f'{dir_data}/proteomes').mkdir(parents=True, exist_ok=True)
    Path(f'{dir_data}/plots').mkdir(parents=True, exist_ok=True)
    Path(f'{dir_data}/tensors').mkdir(parents=True, exist_ok=True)
    Path(f'{dir_results}').mkdir(parents=True, exist_ok=True)


def load_anage_db(filename: str) -> AnAgeDatabase:
    return AnAgeDatabase(filename, dir_data)


def load_ncbi_db(filename: str) -> NCBIDatabase:
    return NCBIDatabase(filename, dir_data)


def load_esm_model() -> ESM:
    return ESM()


def load_rf(results_filename: str) -> RF:
    return RF(results_filename)


def run_analysis(records_file: str,
                 out_directory: str,
                 species_gene_map: Dict[str, List[str]],
                 rf_params: dict,
                 min_seq: float = 0.8,
                 cov: float = 0.8,
                 cov_mode: int = 0,
                 force_new_mmseqs: bool = False):
    """
    Run full analysis of extracted proteins:
        1. Cluster sequences with provided parameters
        2. For each species create vector with counts of genes in each cluster
        3. Run ``RandomForestRegressor`` to find predictor
    """
    # if all conditions for new run are met, do it and save results to ``results_file``
    if not Path(results_file := f'{out_directory}/results.json').exists() or force_new_mmseqs:
        vectors, clusters = run_mmseqs_pipeline(
            records_file,
            out_directory,
            species_gene_map,
            min_seq,
            cov,
            cov_mode,
            force_new_mmseqs=force_new_mmseqs
        )

        final_data = {
            'clusters': clusters,
            'species': {
                species: {
                    'longevity': anage_db.get_longevity(species),
                    'vec': vectors[species]
                }
                for species in vectors
            }
        }
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=4)

    rf = load_rf(results_file)
    score, score_test = rf.process(rf_params, out_directory, show=True, save=True)

    rf_results = {
        'params': [min_seq, cov, cov_mode],
        'scores': [score, score_test]
    }
    print(f'RF params: ({min_seq}, {cov}, {cov_mode})', f'RF scores: ({score}, {score_test})')
    return rf_results


def analysis_check(records_file: str, species_gene_map: Dict[str, List[str]], out_directory: str):
    """Do Random Forest Regressor analysis for multiple parameters combinations."""
    grid_params = {
        'estimators': [10, 100, 300, 500],
        'depth': [None, 5, 7, 12]
    }
    current_results = []
    for estimators in grid_params['estimators']:
        for depth in grid_params['depth']:
            rf_params = {
                'n_estimators': estimators,
                'max_depth': depth
            }
            results = run_analysis(records_file, out_directory, species_gene_map, rf_params, 0.8, 0.8, 0, False)
            logging.info(f'Analysis done on {count_records(records_file)} '
                         f'proteins from {len(species_gene_map)} species\n'
                         f'RF parameters: n estimators = {estimators}, max depth = {depth}')
            current_results.append({
                'results': results,
                'rf_params': rf_params
            })

    load_and_add_results(f'{out_directory}/check.json', current_results)


def mmseq_check(out_directory: str, min_param: float = 0.1, max_param: float = 0.9, step: float = 0.1):
    """Cluster sequences using multiple parameters combinations."""
    current_results = []
    if not Path(p := f'{out_directory}/check_mmseqs.json').exists():
        for cov_mode in range(3):
            for min_seq in np.arange(min_param, max_param, step):
                for cov in np.arange(min_param, max_param, step):
                    r = run_analysis(seqs_file, ex_dir, species_gene_map, {}, min_seq, cov, 0, True)
                    current_results.append(r)
                with open(p, 'w') as f:
                    json.dump(current_results, f, indent=4)

    all_results = load_and_add_results(f'{out_directory}/check_mmseqs.json', current_results)

    RF.scatter3d(all_results, scores=0)
    RF.scatter3d(all_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=agecrack_ng_description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--filter', type=str, default='repair', help='Filter used for extracting proteins sequences')
    parser.add_argument('--anage', type=str, default=f'{dir_data}/anage_data.txt', help='AnAge database file')
    parser.add_argument('--ncbi', type=str, default=f'{dir_data}/eukaryotes.txt', help='NCBI eukaryotes database file')
    parser.add_argument('--skip', action='store_true', help='Skip downloading and extracting part')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity')
    args = parser.parse_args()

    extract_filter = args.filter
    ex_dir = f'{dir_results}/{extract_filter}'
    seqs_file = f'{ex_dir}/sequences.fasta'
    map_file = f'{ex_dir}/mapping.json'

    load_logger(args.verbose)
    create_dirs_structure()

    if not args.skip:
        anage_db = load_anage_db(args.anage)
        ncbi_db = load_ncbi_db(args.ncbi)
        filenames = download_proteomes_by_names(anage_db.species_full_names, f'{dir_data}/proteomes')

        # # histograms of obtained species
        # anage_db.analyze(filenames)

        if not Path(seqs_file).exists():
            # sequences, species_gene_map = extract_proteins(filenames, extract_filter, ['Homo sapiens'])
            sequences, species_gene_map = extract_proteins(filenames, extract_filter, [])
            save_records(sequences, seqs_file)

            # save mapping
            if not Path(map_file).exists():
                with open(map_file, 'w') as f:
                    json.dump(species_gene_map, f, indent=4)

    # we need to load only mapping file thus sequences file will be used directly
    with open(map_file) as f:
        species_gene_map = json.load(f)

    analysis_check(seqs_file, species_gene_map, ex_dir)

    # esm model for creating images from aa sequences
    # esm_model = load_esm_model()
    # esm_model.process_files(filenames)

    # mmseq_check()
    # using mmseq_check, parameters were estimated: 0.8, 0.8, 0

    # # best parameters
    # 'repair'
    # rf_params = {
    #     'n_estimators': 300,
    #     'max_depth': 100
    # }
    # results = run_analysis(seqs_file, ex_dir, species_gene_map, rf_params, 0.8, 0.8, 0, False)
    # print(f'Analysis done on {count_records(seqs_file)} proteins from {len(species_gene_map)} species')
