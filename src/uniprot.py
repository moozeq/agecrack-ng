import json
import logging
from pathlib import Path
from typing import List

import requests

from src.utils import normalize_species


class Uniprot:
    """Functions calling Uniprot API to retrieve proteomes or their IDs."""
    ok = 0
    checked = 0

    @staticmethod
    def get_proteome_id_by_organism(organism: str) -> str:
        """Get ID of best proteome for specified organism."""
        query = f'query=organism:{organism}+taxonomy:chordata&format=list&sort=score'
        url = f'https://www.uniprot.org/proteomes/?{query}'
        try:
            ids = requests.get(url)
            ids = ids.content.decode()
            if not ids:
                raise Exception('empty list')
            pid = ids.splitlines()[0]
            if not pid.startswith('UP'):
                raise Exception(f'wrong pid = {pid}')
            Uniprot.ok += 1
            Uniprot.checked += 1
            logging.info(f'[{Uniprot.ok}/{Uniprot.checked}] Get proteome ID: {organism} -> {pid}')
            return pid
        except Exception as e:
            Uniprot.checked += 1
            if Uniprot.checked % 100 == 0:
                logging.info(f'Checked: {Uniprot.checked}')
            #logging.error(f'[{Uniprot.checked}] Could not download proteome IDs list for: {organism}, error = {str(e)}')
            return ''

    @staticmethod
    def download_proteomes_ids(tax: str, out: str) -> str:
        """Download all proteomes IDs for specific tax family."""
        query = f'query=taxonomy:{tax}&format=tab&sort=score'
        url = f'https://www.uniprot.org/proteomes/?{query}'
        try:
            if Path(out).exists():
                return out
            ids = requests.get(url)
            ids = ids.content.decode()
            if not ids:
                raise Exception('empty list')
            logging.info(f'Downloaded proteomes IDs list: {len(ids.splitlines()) - 1}')
            with open(out, 'w') as fp:
                fp.write(ids)
            return out
        except Exception as e:
            logging.error(f'Could not download proteomes IDs list for: {tax}, error = {str(e)}')
            return ''

    @staticmethod
    def download_proteome(pid: str, org: str, o_dir: str):
        """Download proteome using proteome Uniprot ID."""
        query = f'query=proteome:{pid}&format=fasta&compress=no'
        url = f'https://www.uniprot.org/uniprot/?{query}'
        try:
            if Path(pfile := f'{o_dir}/{normalize_species(org)}.fasta').exists():
                return pfile
            ids = requests.get(url)
            ids = ids.content.decode()
            if not ids:
                raise Exception('empty proteome')
            logging.info(f'Downloaded proteome for: {org}')
            with open(pfile, 'w') as fp:
                fp.write(ids)
            return pfile
        except Exception as e:
            logging.error(f'Could not download proteome for: {org}, error = {str(e)}')
            return ''


def download_proteomes_by_names(names: List[str], fastas_out: str, limit: int = 100000, ids_file: str = 'proteomes.json') -> List[str]:
    """Providing list of organisms names, try to download proteomes with max limit."""
    logging.info(f'Downloading proteomes for {len(names)} organisms')

    if not Path(ids_file).exists():
        pids = {
            pid: org
            for org in names
            if (
                not Path(f'{fastas_out}/{normalize_species(org)}.fasta').exists() and
                (pid := Uniprot.get_proteome_id_by_organism(org))
            )
        }
        with open(ids_file, 'w') as f:
            json.dump(pids, f, indent=4)
    else:
        with open(ids_file) as f:
            pids = json.load(f)

    logging.info(f'Found proteomes IDs for {len(pids)}/{len(names)}')

    proteomes_files = {
        org: str(prot_file)
        for org in names
        if (prot_file := Path(f'{fastas_out}/{normalize_species(org)}.fasta')).exists()
    }

    if not pids and not proteomes_files:
        raise Exception('No proteome IDs loaded')

    if len(proteomes_files) >= limit:
        logging.info(f'Used only local proteomes (limit = {limit}): {len(proteomes_files)}/{len(names)}')
        return list(proteomes_files.values())[:limit]

    logging.info(f'Translated organisms names to proteomes IDs: {len(pids) + len(proteomes_files)}/{len(names)}')
    for i, (pid, org) in enumerate(pids.items()):
        if (
            len(proteomes_files) < limit and
            org not in proteomes_files and
            (prot_file := Uniprot.download_proteome(pid, org, fastas_out))
        ):
            proteomes_files[org] = prot_file

    logging.info(f'Downloaded proteomes for: {len(proteomes_files)}/{len(names)}')
    return list(proteomes_files.values())
