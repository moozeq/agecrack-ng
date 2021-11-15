import csv
import json
import logging
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Dict

import numpy as np
import requests
from matplotlib import pyplot as plt


@dataclass
class AnAgeEntry:
    """Mapping for each AnAge Entry to Python object"""

    species_kingdom: str
    species_phylum: str
    species_class: str
    species_order: str
    species_full_name: str
    species_name: str
    longevity: float
    origin: str
    quality: str

    @classmethod
    def read_vertebrate_from_entry(cls, entry: List[str]):
        try:
            args = (species_kingdom,
                    species_phylum,
                    species_class,
                    species_order,
                    species_full_name,
                    species_name,
                    longevity,
                    origin,
                    quality) = (
                entry[1],
                entry[2],
                entry[3],
                entry[4],
                f'{entry[6]} {entry[7]}',
                entry[8],
                float(entry[20]),
                entry[22],
                entry[24]
            )
            if species_phylum != 'Chordata':
                raise Exception('Entry not in "Chordata" phylum')
            return cls(*args)
        except (ValueError, KeyError, IndexError, Exception) as e:
            logging.debug(f'Could not create AnAgeEntry, error: {str(e)}')
            return None

    @classmethod
    def read_vertebrate_from_dict(cls, entry: Dict):
        try:
            return cls(**entry)
        except (ValueError, KeyError, IndexError, Exception) as e:
            logging.debug(f'Could not create AnAgeEntry, error: {str(e)}')
            return None


class AnAgeDatabase:
    """Mapping for AnAge database to Python object.

    It only supports Vertebrates but can be easily extended to other phyla.

    Attributes:
        vertebrates (List[AnAgeEntry]): List of all vertebrates entries.
        vertebrates_longevity (Dict[str, AnAgeEntry]): Map species full name to its longevity.
        vertebrates_mapping (Dict[str, AnAgeEntry]): Map species full name to its AnAgeEntry.
    """

    link = 'https://genomics.senescence.info/species/dataset.zip'
    zip_file = 'data/dataset.zip'
    db_file = 'anage_data.txt'

    # human-readable filters for phylogenetic
    filters = {
        'birds': ['Aves'],
        'mammals': ['Mammalia'],
        'reptiles': ['Reptilia'],
        'amphibians': ['Amphibia'],
        'fish': [
            'Teleostei',
            'Chondrichthyes',
            'Cephalaspidomorphi',
            'Chondrostei',
            'Holostei',
            'Dipnoi',
            'Cladistei',
            'Coelacanthi'
        ],
        'all': [
            'Aves',
            'Mammalia',
            'Reptilia',
            'Amphibia',
            'Teleostei',
            'Chondrichthyes',
            'Cephalaspidomorphi',
            'Chondrostei',
            'Holostei',
            'Dipnoi',
            'Cladistei',
            'Coelacanthi'
        ]
    }

    def __init__(self, filename: str, output_dir: str = '.', ncbi_db: 'NCBIDatabase' = None):
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
            self.vertebrates_longevity = {
                entry_obj.species_full_name: entry_obj.longevity
                for entry_obj in self.vertebrates
            }
            self.vertebrates_mapping: Dict[str, AnAgeEntry] = {
                entry_obj.species_full_name: entry_obj
                for entry_obj in self.vertebrates
            }
            logging.info(f'Loaded {len(self.vertebrates)} vertebrates from file: {filename}')
            logging.info(f'Vertebrates classes: {self.species_classes}')

        self.ncbi_db = ncbi_db
        self.output_dir = output_dir

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

    def get_longevity(self, species_full_name: str) -> Optional[float]:
        """Get longevity for specific species, otherwise return None."""
        return self.vertebrates_longevity.get(species_full_name, None)

    def get_vertebrate(self, species_full_name: str) -> Optional[AnAgeEntry]:
        """Get specific vertebrate entry, otherwise return None."""
        return self.vertebrates_mapping.get(species_full_name, None)

    def filter(self, filter_name: str) -> list:
        """Filter vertebrates based on predefined filters."""
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

    def _match_ncbi(self):
        """Match entries from AnAge database with entries from NCBI database.

        It can be used for retrieving full genomes of species, currently not in use.
        """
        if not self.ncbi_db:
            logging.warning(f'No NCBI database linked to AnAge database')
            return None

        def name_in_ncbi(n):
            for n_entry in self.ncbi_db.animals:
                if n in n_entry.species_name:
                    return n_entry
            return None

        # two list for entries, one for AnAge entries,
        # second one for NCBI entries
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
        with open('data/ncbi_species_mapping.json', 'w') as f:
            json.dump(species, f, indent=4)

    def _prepare_longevity_hist(self, vert_classes: bool = False):
        """Prepare longevity histogram, or histograms for vertebrates species."""

        def ax_longevity_hist(ax, sp, f):
            ax.hist(sp, np.arange(min(sp), max(sp) + 1, 1), color='grey')
            ax.set_xlim((0, 100))
            ax.set_ylim((0, 180))
            ax.set_title(f'{f} ({len(sp)}), longevity range ({min(sp)}, {max(sp)})')
            ax.axvline(mean(sp), color='r', linestyle='--', label=f'mean {mean(sp):.2f}')
            ax.axvline(median(sp), color='g', linestyle='--', label=f'median {median(sp):.2f}')
            ax.legend()

        def longevity_hists():
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
            axes = [ax1, ax2, ax3, ax4, ax5, ax6]
            plt.tight_layout()
            for i, f in enumerate(AnAgeDatabase.filters):
                vertebrates_filtered = self.filter(f)
                sp = [int(entry.longevity) for entry in vertebrates_filtered]
                ax_longevity_hist(axes[i], sp, f)

        def longevity_hists_vert_class():
            from matplotlib.pyplot import figure

            figure(figsize=(8, 16), dpi=600)

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(nrows=6,
                                                                                                            ncols=2)
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
            plt.tight_layout()
            for i, f in enumerate(AnAgeDatabase.filters['all']):
                sp = [int(entry.longevity) for entry in self.vertebrates if entry.species_class == f]
                if not sp:
                    continue
                ax_longevity_hist(axes[i], sp, f)

        return longevity_hists_vert_class() if vert_classes else longevity_hists()

    def plt_longevity_hist(self, names: List[str]):
        """Plot longevity histograms."""
        names = [name[len('data/proteomes/'):-len('.fasta')].replace('_', ' ') for name in names]
        self._prepare_longevity_hist()
        self.vertebrates = [entry for entry in self.vertebrates if entry.species_full_name in names]
        self._prepare_longevity_hist()
        self._prepare_longevity_hist(True)
        plt.show()

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
