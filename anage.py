import csv
import logging
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from statistics import mean, median
from typing import List

import numpy as np
import requests
from matplotlib import pyplot as plt

from ncbi import NCBIDatabase
from uniprot import download_proteomes_by_names


@dataclass
class AnAgeEntry:
    species_kingdom: str
    species_phylum: str
    species_class: str
    species_full_name: str
    species_name: str
    longevity: float
    origin: str
    quality: str

    @classmethod
    def read_vertebrate_from_entry(cls, entry: List[str]):
        try:
            (species_kingdom,
             species_phylum,
             species_class,
             species_full_name,
             species_name,
             longevity,
             origin,
             quality) = (
                entry[1],
                entry[2],
                entry[3],
                f'{entry[6]} {entry[7]}',
                entry[8],
                float(entry[20]),
                entry[22],
                entry[24]
            )
            if species_phylum != 'Chordata':
                raise Exception('Entry not in "Chordata" phylum')
            return cls(species_kingdom, species_phylum, species_class, species_full_name, species_name, longevity, origin, quality)
        except (ValueError, KeyError, IndexError, Exception) as e:
            logging.debug(f'Could not create AnAgeEntry, error: {str(e)}')
            return None


class AnAgeDatabase:
    link = 'https://genomics.senescence.info/species/dataset.zip'
    zip_file = 'data/dataset.zip'
    db_file = 'anage_data.txt'
    filters = {
        'birds': ['Aves'],
        'mammals': ['Mammalia'],
        'reptiles': ['Reptilia'],
        'amphibians': ['Amphibia'],
        'fish': ['Teleostei', 'Chondrichthyes', 'Cephalaspidomorphi', 'Chondrostei', 'Holostei', 'Dipnoi', 'Cladistei',
                 'Coelacanthi'],
        'all': ['Aves', 'Mammalia', 'Reptilia', 'Amphibia', 'Teleostei', 'Chondrichthyes', 'Cephalaspidomorphi',
                'Chondrostei', 'Holostei', 'Dipnoi', 'Cladistei', 'Coelacanthi']
    }

    def __init__(self, filename: str, output_dir: str = '.'):
        if not Path(filename).exists():
            AnAgeDatabase._download_anage_database(output_dir)

        with open(filename) as f:
            reader = csv.reader(f, delimiter='\t')
            self.header = next(reader)
            self.vertebrates = [
                entry_obj
                for entry in reader
                if (entry_obj := AnAgeEntry.read_vertebrate_from_entry(entry))
            ]
            logging.info(f'Loaded {len(self.vertebrates)} vertebrates from file: {filename}')
            logging.info(f'Vertebrates classes: {self.species_classes}')

        if new_classes := (set(self.species_classes) - set(chain(*AnAgeDatabase.filters.values()))):
            logging.warning(f'New classes of species in database: {new_classes}')

    @property
    def species_classes(self):
        return Counter(entry.species_class for entry in self.vertebrates)

    @property
    def species_full_names(self):
        return [entry.species_full_name for entry in self.vertebrates]

    @property
    def species_names(self):
        return [entry.species_name for entry in self.vertebrates]

    def filter(self, filter_name: str) -> list:
        """Filter vertebrates based on predefined filters"""
        try:
            filter_classes = AnAgeDatabase.filters[filter_name]
            return [
                entry
                for entry in self.vertebrates
                if entry.species_class in filter_classes
            ]
        except KeyError as e:
            logging.warning(f'Wrong filter provided: {filter_name}, error: {str(e)}')
            return []

    def longevity_hists(self):
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        plt.tight_layout()
        for i, f in enumerate(AnAgeDatabase.filters):
            vertebrates_filtered = self.filter(f)
            sp = [int(entry.longevity) for entry in vertebrates_filtered]
            ax = axes[i]
            ax.hist(sp, np.arange(min(sp), max(sp) + 1, 1), color='grey')
            ax.set_xlim((0, 100))
            ax.set_ylim((0, 180))
            ax.set_title(f'{f} ({len(sp)}), longevity range ({min(sp)}, {max(sp)})')
            ax.axvline(mean(sp), color='r', linestyle='--', label=f'mean {mean(sp):.2f}')
            ax.axvline(median(sp), color='g', linestyle='--', label=f'median {median(sp):.2f}')
            ax.legend()

    def analyze(self, ncbi_db: NCBIDatabase):
        names = download_proteomes_by_names(self.species_full_names, 'data/proteomes')
        names = [name[len('data/proteomes/'):-len('.fasta')].replace('_', ' ') for name in names]
        self.longevity_hists()
        self.vertebrates = [entry for entry in self.vertebrates if entry.species_full_name in names]
        self.longevity_hists()
        plt.show()
        sys.exit(0)

        def name_in_ncbi(n):
            for n_entry in ncbi_db.animals:
                if n in n_entry.species_name:
                    return n_entry
            return None

        c = []
        n = []
        for entry in self.vertebrates:
            if ncbi_entry := name_in_ncbi(entry.species_full_name):
                c.append(entry)
                n.append(ncbi_entry)

        logging.info(f'Matched {len(c)}/{len(self.species_full_names)} with NCBI database!')
        logging.info(f'Classes = {Counter([ce.species_class for ce in c])}')

        size = sum(ne.genome_size for ne in n)
        logging.info(f'Size for all genomes = {size:.2f}MB')

        species = {ce.species_full_name: ne.species_name for ce, ne in zip(c, n)}
        with open('data/species.json', 'w') as f:
            import json
            json.dump(species, f, indent=4)

    @staticmethod
    def _download_anage_database(output_dir: str):
        logging.info(f'Download AnAge database from: {AnAgeDatabase.link}')
        try:
            if not Path(AnAgeDatabase.zip_file).exists():
                resp = requests.get(AnAgeDatabase.link)
                resp.raise_for_status()
                with open(AnAgeDatabase.zip_file, 'wb') as f:
                    f.write(resp.content)
                logging.info(f'Database downloaded and stored at: {AnAgeDatabase.zip_file}')

            with zipfile.ZipFile(AnAgeDatabase.zip_file) as f:
                f.extract(AnAgeDatabase.db_file, output_dir)
            logging.info(f'Database file extracted under "data/" directory: {AnAgeDatabase.db_file}')

            if not Path(f'{output_dir}/{AnAgeDatabase.db_file}').exists():
                raise Exception(f'AnAge database file not found')

        except (requests.ConnectionError, requests.HTTPError, Exception) as e:
            logging.error(f'Could not download AnAge database, error: {str(e)}')
            sys.exit(1)
