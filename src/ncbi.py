import csv
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests


@dataclass
class NCBIEntry:
    species_name: str
    species_kingdom: str
    species_class: str
    genome_size: float
    gc_ratio: float
    accession: str
    genes: int
    proteins: int
    status: str

    @classmethod
    def read_animal_from_entry(cls, entry: List[str]):
        try:
            (species_name,
             species_kingdom,
             species_class,
             genome_size,
             gc_ratio,
             accession,
             genes,
             proteins,
             status) = (
                entry[0],
                entry[4],
                entry[5],
                float(entry[6]),
                float(entry[7]),
                entry[8],
                entry[12],
                entry[13],
                entry[16]
            )

            try:
                genes = int(genes)
            except ValueError:
                genes = 0

            try:
                proteins = int(proteins)
            except ValueError:
                proteins = 0

            if species_kingdom != 'Animals':
                raise Exception('Entry not in "Animals" kingdom')
            return cls(species_name, species_kingdom, species_class, genome_size, gc_ratio, accession, genes, proteins, status)
        except (ValueError, KeyError, IndexError, Exception) as e:
            logging.debug(f'Could not create NCBIEntry, error: {str(e)}')
            return None


class NCBIDatabase:
    link = 'https://ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/eukaryotes.txt'
    db_file = 'eukaryotes.txt'

    def __init__(self, filename: str, output_dir: str = '.'):
        if not Path(filename).exists():
            NCBIDatabase._download_ncbi_database(output_dir)

        with open(filename) as f:
            reader = csv.reader(f, delimiter='\t')
            self.header = next(reader)
            self.animals = [
                entry_obj
                for entry in reader
                if (entry_obj := NCBIEntry.read_animal_from_entry(entry))
            ]
            logging.info(f'Loaded {len(self.animals)} animals from file: {filename}')
            logging.info(f'Animals classes: {self.species_classes}')

    @property
    def species_classes(self):
        return Counter(entry.species_class for entry in self.animals)

    @property
    def species_full_names(self):
        return [entry.species_name for entry in self.animals]

    @staticmethod
    def _download_ncbi_database(output_dir: str):
        logging.info(f'Download NCBI eukaryotes database from: {NCBIDatabase.link}')
        try:
            db_file = f'{output_dir}/{NCBIDatabase.db_file}'
            if not Path(db_file).exists():
                resp = requests.get(NCBIDatabase.link)
                resp.raise_for_status()
                with open(db_file, 'wb') as f:
                    f.write(resp.content)
                logging.info(f'Database downloaded and stored at: {NCBIDatabase.db_file}')

        except (requests.ConnectionError, requests.HTTPError, Exception) as e:
            logging.error(f'Could not download NCBI database, error: {str(e)}')
            sys.exit(1)
