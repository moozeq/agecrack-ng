#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Type

import numpy as np

from anage import AnAgeDatabase, AnAgeEntry
from mmseq import run_mmseqs_pipeline, MmseqConfig
from models import Model, RF, EN, ANN, ModelsConfig
from ncbi import NCBIDatabase
from prot_encode import ESM
from uniprot import download_proteomes_by_names
from utils import (extract_proteins, save_records, count_records, load_and_add_results, plot_ontology_stats,
                   map_clusters_to_descs, map_ids_to_descs, )

DIR_DATA = 'data'
DIR_RESULTS = 'results'

AGECRACK_NG_DESCRIPTION = """
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
    Path(f'{DIR_DATA}/proteomes').mkdir(parents=True, exist_ok=True)
    Path(f'{DIR_RESULTS}').mkdir(parents=True, exist_ok=True)


def load_anage_db(filename: str) -> AnAgeDatabase:
    return AnAgeDatabase(filename, DIR_DATA)


def load_ncbi_db(filename: str) -> NCBIDatabase:
    return NCBIDatabase(filename, DIR_DATA)


def load_esm_model() -> ESM:
    return ESM()


def run_analysis(records_file: str,
                 out_directory: str,
                 species_map: Dict[str, dict],
                 params: dict,
                 class_filter: str,
                 ontology_file: str,
                 mmseq_config: MmseqConfig,
                 models_config: ModelsConfig,
                 anage_db: AnAgeDatabase = None,
                 model: Type[Model] = EN,
                 results_dict: dict = None):
    """
    Run full analysis of extracted proteins:
        1. Cluster sequences with provided parameters
        2. For each species create vector with counts of genes in each cluster
        3. Run ``Regressor`` to find predictor
    """
    # if all conditions for new run are met, do it and save results to ``results_file``
    if (
            not Path(results_file := f'{out_directory}/results.json').exists()
            or mmseq_config.force_new_mmseqs
            or mmseq_config.reload_mmseqs
    ):
        vectors, clusters = run_mmseqs_pipeline(
            records_file,
            species_map,
            mmseq_config,
            out_directory
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

    if not Path(ontology_file).exists() or mmseq_config.force_new_mmseqs or mmseq_config.reload_mmseqs:
        genes_to_descs = map_ids_to_descs(records_file)
        cls_to_descs = map_clusters_to_descs(mmseq_config.clusters_file, genes_to_descs)
        with open(ontology_file, 'w') as f:
            json.dump(cls_to_descs, f)

    if results_dict:
        m = model.from_dict(results_dict, species_map, class_filter, ontology_file)
    else:
        m = model.from_file(results_file, species_map, class_filter, ontology_file)

    score, score_test = m.process(params, out_directory, models_config)

    m_results = {
        'params': [mmseq_config.min_seq_id, mmseq_config.c, mmseq_config.cov_mode],
        'scores': [score, score_test]
    }
    logging.info(f'Model params: ({mmseq_config.min_seq_id}, {mmseq_config.c}, {mmseq_config.cov_mode}) '
                 f'Model scores: ({score:.2f}, {score_test:.2f})')
    return m_results, m


def analysis_check_rf(records_file: str,
                      species_map: Dict[str, dict],
                      class_filter: str,
                      proteins_count: str,
                      grid_params: dict,
                      ontology_file: str,
                      mmseq_config: MmseqConfig,
                      models_config: ModelsConfig,
                      anage_db: AnAgeDatabase,
                      out_directory: str):
    """Do Random Forest Regressor analysis for multiple parameters combinations."""
    current_results = []

    # read model from file to get it into memory, speeding up multiple calls
    results_dict = Model.read_results_file(res) if Path(res := f'{out_directory}/results.json').exists() else {}

    for estimators in grid_params['estimators']:
        for depth in grid_params['depth']:
            rf_params = {
                'n_estimators': estimators,
                'max_depth': depth
            }
            results, rf = run_analysis(records_file,
                                       out_directory,
                                       species_map,
                                       rf_params,
                                       class_filter,
                                       ontology_file,
                                       mmseq_config,
                                       models_config,
                                       anage_db,
                                       RF,
                                       results_dict)

            logging.info(f'Analysis done on {proteins_count} '
                         f'proteins from {len(species_map)} species\n'
                         f'RF parameters: n estimators = {estimators}, max depth = {depth}')
            current_results.append({
                'results': results,
                'rf_params': rf_params
            })

    load_and_add_results(f'{out_directory}/check.json', current_results)


def analysis_check_en(records_file: str,
                      species_map: Dict[str, dict],
                      class_filter: str,
                      proteins_count: str,
                      grid_params: dict,
                      ontology_file: str,
                      mmseq_config: MmseqConfig,
                      models_config: ModelsConfig,
                      anage_db: AnAgeDatabase,
                      out_directory: str):
    """Do Elastic Net Regressor analysis for multiple parameters combinations."""

    current_results = []
    results_dict = Model.read_results_file(res) if Path(res := f'{out_directory}/results.json').exists() else {}
    for alpha in grid_params['alpha']:
        for l1_ratio in grid_params['l1_ratio']:
            params = {
                'alpha': alpha,
                'l1_ratio': l1_ratio
            }
            results, en = run_analysis(records_file,
                                       out_directory,
                                       species_map,
                                       params,
                                       class_filter,
                                       ontology_file,
                                       mmseq_config,
                                       models_config,
                                       anage_db,
                                       EN,
                                       results_dict)

            logging.info(f'Analysis done on {proteins_count} '
                         f'proteins from {len(species_map)} species\n'
                         f'Model parameters: {str(params)}')
            current_results.append({
                'results': results,
                'params': params
            })

    load_and_add_results(f'{out_directory}/check.json', current_results)


def analysis_check_ann(records_file: str,
                       species_map: Dict[str, dict],
                       class_filter: str,
                       proteins_count: str,
                       grid_params: dict,
                       ontology_file: str,
                       mmseq_config: MmseqConfig,
                       models_config: ModelsConfig,
                       anage_db: AnAgeDatabase,
                       out_directory: str):
    """Do Neural Network Regressor analysis for multiple parameters combinations. Currently not in use."""
    results, ann = run_analysis(records_file, out_directory, species_map, {}, class_filter, ontology_file, mmseq_config,
                                models_config, anage_db, ANN)

    logging.info(f'Analysis done on {proteins_count} '
                 f'proteins from {len(species_map)} species')


def mmseq_check(records_file: str,
                species_map: Dict[str, dict],
                out_directory: str,
                min_param: float = 0.1,
                max_param: float = 0.9,
                step: float = 0.1):
    """Cluster sequences using multiple parameters combinations."""
    current_results = []
    mmseq_config = MmseqConfig(f'{out_directory}/clusters.json')

    if not Path(p := f'{out_directory}/check_mmseqs.json').exists():
        # coverage mode is an int: 0, 1 or 2
        for cov_mode in range(3):
            mmseq_config.cov_mode = cov_mode
            for min_seq_id in np.arange(min_param, max_param, step):
                mmseq_config.min_seq_id = min_seq_id
                for c in np.arange(min_param, max_param, step):
                    mmseq_config.c = c
                    r = run_analysis(records_file, ex_dir, species_map, {}, '', mmseq_config)
                    current_results.append(r)
                with open(p, 'w') as f:
                    json.dump(current_results, f, indent=4)

    all_results = load_and_add_results(f'{out_directory}/check_mmseqs.json', current_results)

    Model.scatter3d(all_results, scores=0)
    Model.scatter3d(all_results)


def load_grid_params(mode: str, model: str):
    # for full/ontology analysis do multiple parameters from file
    if mode in ['full', 'ontology']:
        with open('grid_params.json') as f:
            predef_grid_params = json.load(f)
    # best estimated parameters for predictors,
    # wrapped in list for compatibility
    else:
        predef_grid_params = {
            'rf': {
                'estimators': [300],
                'depth': [18]
            },
            'en': {
                'alpha': [0.01],
                'l1_ratio': [0.7]
            },
            'ann': {}
        }
    return predef_grid_params[model]


def load_mmseq_config(cluster_file: str, reload: bool, mmseq_force: bool, mmseq_threshold: int):
    # create and update ``MmseqConfig``
    mmseq_config = MmseqConfig(cluster_file)
    mmseq_config.reload_mmseqs = reload
    mmseq_config.force_new_mmseqs = mmseq_force
    mmseq_config.cluster_count_threshold = mmseq_threshold
    return mmseq_config


def load_models_config(models_reuse: bool, plots_show: bool, plots_annotate: bool):
    # create and upadte ``PlotsConfig``
    models_config = ModelsConfig(models_reuse, plots_show, plots_annotate)
    return models_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=AGECRACK_NG_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode',
                        type=str, default='predictor',
                        choices=['full', 'predictor', 'ontology', 'vectors', 'mmseqs-estimation'],
                        help='Select mode for running the program, '
                             '"predictor" gives single best predictor for longevity based on predefined parameters, '
                             '"ontology" runs full analysis for clusters ontology and correlation with longevity, '
                             '"vectors" produces additional visualization of species genes vectors, '
                             '"mmseqs-estimation" produces additional plots for mmseqs params estimation, '
                             '"full" runs whole ontology and vectors analysis')
    parser.add_argument('--model',
                        type=str, default='en', choices=['rf', 'en', 'ann'],
                        help='ML model')
    parser.add_argument('--filters',
                        nargs='+', default=[''],
                        help='Filters used for extracting proteins sequences, examples in "filters.json" file, '
                             'when list with empty string is provided (`[\'\']`) - no filtering is applied')
    parser.add_argument('--filter-class',
                        type=str,
                        help='Filter species from specific phylo class (e.g. "Mammalia")')
    parser.add_argument('--exclude',
                        nargs='+', default=[],
                        help='List of excluded species from analysis')
    parser.add_argument('--extract-threshold',
                        type=int, default=1,
                        help='Filter out species with count of genes below threshold')
    parser.add_argument('--anage',
                        type=str, default=f'{DIR_DATA}/anage_data.txt',
                        help='AnAge database file')
    parser.add_argument('--ncbi',
                        type=str, default=f'{DIR_DATA}/eukaryotes.txt',
                        help='NCBI eukaryotes database file')
    parser.add_argument('--skip',
                        action='store_true',
                        help='Skip downloading and extracting part, use it to speed up when trying different models, '
                             '(option is omitted when running mmseq)')
    parser.add_argument('--count-proteins',
                        action='store_true',
                        help='Count proteins for proper log messages (can impact performance greatly)')
    parser.add_argument('--reload',
                        action='store_true',
                        help='Reload produced files, set when changing thresholds')
    parser.add_argument('--models-reuse',
                        action='store_true',
                        help='Reuse ML models from files if exist')
    parser.add_argument('--models-plots-show',
                        action='store_true',
                        help='Show plots for each model')
    parser.add_argument('--models-plots-annotate',
                        action='store_true',
                        help='Annotate points on models plots with species names')
    parser.add_argument('--mmseq-threshold',
                        type=int, default=0,
                        help='Clusters under strength of this threshold will be filter out')
    parser.add_argument('--mmseq-force',
                        action='store_true',
                        help='Force re-running mmseq')
    parser.add_argument('--plot-anage-hists',
                        action='store_true',
                        help='Plot AnAge database histograms')
    parser.add_argument('-v', '--verbose',
                        action='count', default=0,
                        help='Increase verbosity')
    args = parser.parse_args()

    load_logger(args.verbose)
    create_dirs_structure()

    """
    Ideas:
        - [ ] filter proteomes under 1k proteins
        - [ ] PCA
        - [ ] classifier for taxons based on vectors
    """

    for extract_filter in args.filters:
        ex_dir = f'{DIR_RESULTS}/{extract_filter}' if extract_filter else f'{DIR_RESULTS}/_nofilter'
        seqs_file = f'{ex_dir}/sequences.fasta'
        map_file = f'{ex_dir}/mapping.json'
        cluster_file = f'{ex_dir}/clusters.json'
        ontology_file = f'{ex_dir}/ontology.json'

        # download and load AnAge database
        anage_db = None
        if not args.skip or args.mmseq_force or args.reload:
            anage_db = load_anage_db(args.anage)
            ncbi_db = load_ncbi_db(args.ncbi)
            filenames = download_proteomes_by_names(anage_db.species_full_names, f'{DIR_DATA}/proteomes')

            # histograms of obtained species
            if args.plot_anage_hists:
                anage_db.plt_longevity_hist(filenames)

            if not Path(seqs_file).exists() or args.mmseq_force or args.reload:
                sequences, sp_gene_map = extract_proteins(filenames, extract_filter, args.extract_threshold,
                                                          args.exclude)
                save_records(sequences, seqs_file)

                sp_map = {
                    sp_name: {
                        'genes': sp_genes,
                        'AnAgeEntry': anage_db.vertebrates_mapping[sp_name].__dict__
                    }
                    for sp_name, sp_genes in sp_gene_map.items()
                }

                # save species mapping
                if not Path(map_file).exists() or args.mmseq_force or args.reload:
                    with open(map_file, 'w') as f:
                        json.dump(sp_map, f, indent=4)

        # we need to load only mapping file thus sequences file will be used directly
        with open(map_file) as f:
            sp_map = json.load(f)
            sp_map = {
                sp_name: {
                    'genes': sp_dict['genes'],
                    'AnAgeEntry': AnAgeEntry.read_vertebrate_from_dict(sp_dict['AnAgeEntry'])
                }
                for sp_name, sp_dict in sp_map.items()
            }

        # count proteins if specified (impact performance)
        records_count = count_records(seqs_file) if args.count_proteins else '- (counting skipped)'

        analysis_funcs = {
            'rf': analysis_check_rf,
            'en': analysis_check_en,
            'ann': analysis_check_ann
        }

        grid_params = load_grid_params(args.mode, args.model)
        mmseq_config = load_mmseq_config(cluster_file, args.reload, args.mmseq_force, args.mmseq_threshold)
        models_config = load_models_config(args.models_reuse, args.models_plots_show, args.models_plots_annotate)

        # run proper function based on selected model
        analysis_funcs[args.model](
            seqs_file,
            sp_map,
            args.filter_class,
            records_count,
            grid_params,
            ontology_file,
            mmseq_config,
            models_config,
            anage_db,
            ex_dir
        )

        # plot vectors using ``results.json`` from mmseq
        if args.mode in ['full', 'vectors']:
            model = Model.from_file(f'{ex_dir}/results.json', sp_map, args.filter_class, ontology_file)
            model.visualize_data(sp_map, extract_filter, ex_dir)

        # using mmseq_check, parameters were estimated: 0.8, 0.8, 0
        if args.mode in ['mmseqs-estimation']:
            mmseq_check(seqs_file, sp_map, ex_dir)

        if args.mode in ['full', 'ontology']:
            plot_ontology_stats(f'{ex_dir}/ontology', ontology_file, f'{ex_dir}/analysis.json',
                                f'{ex_dir}/analysis.png')
