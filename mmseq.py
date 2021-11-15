import csv
import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class MmseqConfig:
    """Mapping mmseqs2 options to Python object."""

    # file for clusters
    clusters_file: str

    # directly passed to ``mmseqs`` (default values are estimated for this project)
    c: float = 0.8  # List matches above this fraction of aligned (covered) residues
    cov_mode: int = 0  # 0: coverage of query and target; 1: coverage of target; 2: coverage of query
    min_seq_id: float = 0.8  # List matches above this sequence identity (for clustering)

    # config for running ``mmseqs``
    force_new_mmseqs: bool = False  # force running ``mmseqs`` even if results files exist
    reload_mmseqs: bool = False  # reload ``mmseqs`` files, when trying different thresholds
    cluster_count_threshold: int = 0  # threshold for strength of cluster under which they are filtered out


def _run_mmseqs(records_file: str,
                mmseqs_config: MmseqConfig,
                out_directory: str):
    """Run mmseqs on directory with filtered proteomes."""
    cluster_tsv = f'{out_directory}/clusterRes_cluster.tsv'
    if not mmseqs_config.force_new_mmseqs and Path(cluster_tsv).exists():
        return cluster_tsv
    subprocess.call(
        f'mmseqs '
        f'easy-cluster {records_file} {out_directory}/clusterRes {out_directory}/mmseqs '
        f'--min-seq-id {mmseqs_config.min_seq_id} '
        f'-c {mmseqs_config.c} '
        f'--cov-mode {mmseqs_config.cov_mode}',
        shell=True
    )
    if not Path(cluster_tsv).exists():
        raise Exception(f'`mmseqs` probably failed. No cluster tsv file: {cluster_tsv}')

    return cluster_tsv


def _parse_mmseqs_file(mmseqs_cluster_tsv: str,
                       mmseqs_config: MmseqConfig,
                       out_file: str) -> dict:
    """Parse mmseqs results to json format."""
    if Path(out_file).exists() and not (mmseqs_config.force_new_mmseqs or mmseqs_config.reload_mmseqs):
        with open(out_file) as f:
            s_clusters = json.load(f)
    else:
        with open(mmseqs_cluster_tsv) as f:
            reader = csv.reader(f, delimiter='\t')
            clusters = defaultdict(set)
            for row in reader:
                cluster_id, gene_id = row
                clusters[cluster_id].add(gene_id)

        # get only clusters which length is greater than threshold
        s_clusters = {
            k: list(clusters[k])
            for k in sorted(clusters, key=lambda x: len(clusters[x]), reverse=True)
            if len(clusters[k]) > mmseqs_config.cluster_count_threshold
        }
        with open(out_file, 'w') as f:
            json.dump(s_clusters, f, indent=4)

    # convert lists to sets for faster intersections
    s_clusters = {
        cluster: set(genes)
        for cluster, genes in s_clusters.items()
    }
    return s_clusters


def _get_species_clusters_vector(species_genes_ids: List[str], clusters: dict) -> List[int]:
    """For species filtered proteome and clusters from mmseqs, return vector with genes counts."""

    def get_gene_name(rec_id: str) -> str:
        return rec_id.split('|')[1]

    def count_genes_in_cluster(g_ids: set, clus: set) -> int:
        return len(clus.intersection(g_ids))

    genes_ids = set(get_gene_name(gene_id) for gene_id in species_genes_ids)
    vector = []
    for cluster_id, cluster_seqs_ids in clusters.items():
        count = count_genes_in_cluster(genes_ids, cluster_seqs_ids)
        vector.append(count)
    return vector


def run_mmseqs_pipeline(records_file: str,
                        species_map: Dict[str, dict],
                        mmseqs_config: MmseqConfig,
                        out_directory: str) -> (Dict[str, List[int]], List[str]):
    """Run mmseqs pipeline and obtain vectors from sequences"""

    # run mmseqs if forced to new run, or file does not exist
    clusters_tsv = _run_mmseqs(
        records_file,
        mmseqs_config,
        out_directory
    )

    # parse file with clusters (
    clusters = _parse_mmseqs_file(clusters_tsv, mmseqs_config, mmseqs_config.clusters_file)
    vectors = {}

    species_gene_map = {
        sp: set(species_map[sp]['genes'])
        for sp in species_map
    }
    for species, gene_list in species_gene_map.items():
        logging.info(f'Getting vector for: {species}')
        species_vec = _get_species_clusters_vector(gene_list, clusters)
        vectors[species] = species_vec
    return vectors, list(clusters.keys())
