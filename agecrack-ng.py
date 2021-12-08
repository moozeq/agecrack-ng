#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

from src.anage import AnAgeDatabase, AnAgeEntry
from src.logger import load_logger
from src.mmseq import MmseqConfig, mmseq_check, run_mmseqs_pipeline
from src.models import Model, RF, EN, ModelsConfig, ENCV
from src.ncbi import NCBIDatabase
from src.ontology import OntologyConfig, ontology_scores
from src.uniprot import download_proteomes_by_names
from src.utils import extract_proteins, save_records, count_records, CustomArgparseFormatter, save_vectors_file

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


def create_dirs_structure():
    Path(f'{DIR_DATA}/proteomes').mkdir(parents=True, exist_ok=True)
    Path(f'{DIR_RESULTS}').mkdir(parents=True, exist_ok=True)


def load_anage_db(filename: str) -> AnAgeDatabase:
    return AnAgeDatabase(filename, DIR_DATA)


def load_ncbi_db(filename: str) -> NCBIDatabase:
    return NCBIDatabase(filename, DIR_DATA)


def load_grid_params(mode: str, model: str) -> dict:
    with open('grid_params.json') as f:
        predef_grid_params = json.load(f)

    # for full/ontology analysis do multiple parameters
    # for predictor use best estimated parameters
    # if mode not specified in grid params file use predictor
    mode = mode if mode in predef_grid_params else 'predictor'
    return predef_grid_params[mode][model]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=AGECRACK_NG_DESCRIPTION,
        formatter_class=CustomArgparseFormatter
    )
    parser.add_argument('--mode',
                        type=str, default='predictor',
                        choices=['predictor', 'ontology', 'ontology-parse', 'vectors', 'prepare', 'mmseqs-estimation'],
                        help=textwrap.dedent('''\
                        Select mode for running the program:
                             - "predictor" gives single best predictor for longevity based on predefined parameters
                             - "ontology" runs analysis for clusters ontology and correlation with longevity
                             - "ontology-parse" parses files obtained in ontology analysis
                             - "vectors" produces additional visualization of species genes vectors
                             - "prepare" download all data, cluster them and prepare for further analysis
                             - "mmseqs-estimation" produces additional plots for mmseqs params estimation
                        '''))
    parser.add_argument('--model',
                        type=str, default='rf', choices=['rf', 'encv', 'en'],
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
    parser.add_argument('--models-rand',
                        type=int, default=1,
                        help='Random state for splitting data for training and testing')
    parser.add_argument('--models-stratify',
                        action='store_true', dest='models_stratify',
                        help='Try to stratify dataset using bins')
    parser.add_argument('--models-bins',
                        type=int, default=0,
                        help='How many bins for stratifying data, if not specified - number of species divided by 2')
    parser.add_argument('--models-plots-show',
                        action='store_true',
                        help='Show plots for each model')
    parser.add_argument('--models-plots-annotate',
                        action='store_true',
                        help='Annotate points on models plots with species names')
    parser.add_argument('--models-plots-annotate-threshold',
                        type=float, default=0.5,
                        help='Difference between predicted and known lifespan that should be annotated')
    parser.add_argument('--models-plots-clusters-count',
                        type=int, default=30,
                        help='Up to how many most important clusters should be shown on an ontology plot')
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

    for extract_filter in args.filters:
        ex_dir = f'{DIR_RESULTS}/{extract_filter}' if extract_filter else f'{DIR_RESULTS}/_nofilter'
        seqs_file = f'{ex_dir}/sequences.fasta'
        map_file = f'{ex_dir}/mapping.json'
        cluster_file = f'{ex_dir}/clusters.json'
        ontology_file = f'{ex_dir}/ontology.json'
        ontology_dir = f'{ex_dir}/ontology'
        ontology_result = f'{ex_dir}/analysis.json'
        ontology_plot = f'{ex_dir}/analysis.png'
        vectors_file = f'{ex_dir}/vectors.gz'

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
                sequences, sp_gene_map = extract_proteins(filenames,
                                                          extract_filter,
                                                          args.extract_threshold,
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

        # count proteins if specified (may impact performance greatly)
        records_count = count_records(seqs_file) if args.count_proteins else ''

        # load grid params for ML models from file or predefined ones
        grid_params = load_grid_params(args.mode, args.model)

        # prepare proper configs objects
        mmseq_config = MmseqConfig(cluster_file,
                                   args.reload,
                                   args.mmseq_force,
                                   args.mmseq_threshold)

        ontology_config = OntologyConfig(ontology_dir,
                                         ontology_file,
                                         ontology_result,
                                         ontology_plot)

        models_config = ModelsConfig(args.models_reuse,
                                     args.models_plots_show,
                                     args.models_plots_annotate,
                                     args.models_plots_annotate_threshold,
                                     args.models_plots_clusters_count,
                                     args.models_rand,
                                     args.models_bins,
                                     args.models_stratify)

        # select proper model class
        models = {
            'rf': RF,
            'encv': ENCV,
            'en': EN
        }
        selected_model: Model = models[args.model]

        # if all conditions for new run are met, do it and save results to ``results_file``
        if (
                not Path(vectors_file).exists()
                or args.mode in ['prepare']
                or mmseq_config.force_new_mmseqs
                or mmseq_config.reload_mmseqs
        ):
            vectors, clusters = run_mmseqs_pipeline(
                seqs_file,
                sp_map,
                mmseq_config,
                ex_dir
            )
            save_vectors_file(vectors, clusters, anage_db, vectors_file)
            logging.info(f'Clustering and vectorization done, '
                         f'created ({len(vectors)}) species vectors, each with ({len(clusters)}) in length')
            if args.mode in ['prepare']:
                sys.exit(0)

        # run static method from selected model class
        if args.mode in ['predictor', 'ontology', 'vectors']:
            selected_model.analysis_check(
                seqs_file,
                vectors_file,
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

        # plot vectors using ``vectors.gz`` from mmseq
        if args.mode in ['vectors']:
            model = Model(vectors_file, sp_map, args.filter_class, ontology_file)
            model.visualize_vectors(sp_map, extract_filter, ex_dir)

        if args.mode in ['mmseqs-estimation']:
            mmseq_check(seqs_file, sp_map, ex_dir, ontology_file,
                        args.filter_class, grid_params, selected_model,
                        anage_db, mmseq_config, models_config)

        if args.mode in ['ontology', 'ontology-parse']:
            ontology_scores(ontology_config)
