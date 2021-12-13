import argparse
import json
import logging
import timeit
from functools import wraps
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from compress_pickle import compress_pickle
from numpy import ndarray


class CustomArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter,
                              argparse.RawTextHelpFormatter):
    pass


def get_name_from_filename(filename: str) -> str:
    """From filename with fasta proteome, return name of species."""
    return Path(filename).stem.replace('_', ' ')


def normalize_species(sp: str) -> str:
    """Change species name, removing characters which may cause issues in pipeline."""
    not_valid = [' ', '-', '/', '(', ')', '#', ':', ',', ';', '[', ']', '\'', '"', '___', '__']
    for ch in not_valid:
        sp = sp.replace(ch, '_')
    return sp


def load_records(fasta_file: str, keyword: str = '') -> List[SeqRecord]:
    """Load proteins from records which contain specific keyword in description."""
    return [
        record
        for record in SeqIO.parse(fasta_file, 'fasta')
        if not keyword or (keyword.lower() in record.description.lower())
    ]


def load_and_add_results(results_file: str, current_results: list) -> list:
    """Load previous results, add them to current, save and return all."""
    prev_results = []
    if Path(results_file).exists():
        with open(results_file) as f:
            prev_results = json.load(f)
    all_results = prev_results + current_results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    return all_results


def count_records(fasta_file: str, keyword: str = '') -> int:
    """Count records which contain specific keyword in description."""
    return sum(
        True
        for record in SeqIO.parse(fasta_file, 'fasta')
        if not keyword or (keyword.lower() in record.description)
    )


def count_records_dir(dir_with_fastas: str, keyword: str = '') -> int:
    """Count all records under specific directory, which contain specific keyword in description."""
    return sum(count_records(str(fn), keyword) for fn in Path(dir_with_fastas).glob('*.fasta'))


def save_records(records: list, fasta_file: str):
    """Simple saving all records in fasta file."""
    Path(fasta_file).parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, fasta_file, 'fasta')


def extract_proteins(filenames: List[str],
                     extract_filter: str,
                     extract_threshold: int,
                     excluded_species: List[str]) -> (List[str], Dict[str, List[str]]):
    """
    From provided filenames extract proteins based on ``extract_filter`` word and ``extract_threshold``.

    Args:
        filenames : list of filenames from where proteins will be extracted
        extract_filter : filter used for extraction (filter description of genes)
        extract_threshold : filter used for extraction (filter on genes count)
        excluded_species : filter used for excluding species by name

    Returns:
        records : list of proteins sequences extracted from species
        species_gene_map : dict mapping species names to their genes IDs
    """
    records = []
    species = []
    genes_mapping = {}
    for fn in filenames:
        name = get_name_from_filename(fn)
        if name in excluded_species:
            continue
        if data := load_records(fn, extract_filter):
            if len(data) < extract_threshold:
                continue
            records += data
            species.append(name)
            genes_mapping[name] = [rec.id for rec in data]
        logging.info(f'Extracted {len(data):6} {extract_filter} proteins for: {name}')

    logging.info(f'Loaded {len(records)} {extract_filter} proteins data for {len(species)} species')
    return records, genes_mapping


def convert_ys(y, y_p, func):
    """Convert y datasets to normalized form"""
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_p[0], ndarray):
        y_p = np.array([yp[0] for yp in y_p])
    y = func(y) if func else y
    y_p = func(y_p) if func else y_p
    return y, y_p


def save_vectors_file(vectors: dict, clusters: list, anage_db: 'AnAgeDatabase', results_file: str):
    """Save main results file"""
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
    compress_pickle.dump(final_data, results_file)


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.debug(f'[TIMING] Executing "{f.__name__}"...')
        start = timeit.default_timer()
        result = f(*args, **kwargs)
        end = timeit.default_timer()
        logging.debug(f'[TIMING] Finished calculation, elapsed time = {end - start:.2f} seconds')
        return result

    return wrapper
