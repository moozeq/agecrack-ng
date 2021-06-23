import csv
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


def run_mmseqs(records_file: str,
               out_directory: str,
               min_seq: float = 0.8,
               cov: float = 0.8,
               cov_mode: int = 0,
               force_new_mmseqs: bool = False):
    """Run mmseqs on directory with filtered proteomes."""
    cluster_tsv = f'{out_directory}/clusterRes_cluster.tsv'
    if not force_new_mmseqs and Path(cluster_tsv).exists():
        return cluster_tsv
    subprocess.call(
        f'mmseqs '
        f'easy-cluster {records_file} {out_directory}/clusterRes {out_directory}/mmseqs '
        f'--min-seq-id {min_seq} '
        f'-c {cov} '
        f'--cov-mode {cov_mode}',
        shell=True
    )
    if not Path(cluster_tsv).exists():
        raise Exception(f'`mmseqs` probably failed. No cluster tsv file: {cluster_tsv}')

    return cluster_tsv


def parse_mmseqs_file(mmseqs_cluster_tsv: str = 'clusterRes_cluster.tsv',
                      out: str = 'data/repairs_clusters.json') -> dict:
    """Parse mmseqs results to json format."""
    with open(mmseqs_cluster_tsv) as f:
        reader = csv.reader(f, delimiter='\t')
        clusters = defaultdict(list)
        for row in reader:
            cluster_id, gene_id = row
            clusters[cluster_id].append(gene_id)
    s_clusters = {k: clusters[k] for k in sorted(clusters, key=lambda x: len(clusters[x]), reverse=True)}
    with open(out, 'w') as f:
        json.dump(s_clusters, f, indent=4)
    return s_clusters


def get_species_clusters_vector(species_genes_ids: List[str], clusters: dict) -> List[int]:
    """For species filtered proteome and clusters from mmseqs, return vector with genes counts."""

    def get_gene_name(rec_id: str) -> str:
        return rec_id.split('|')[1]

    def count_genes_in_cluster(g_ids: list, clus: list) -> int:
        return sum(clus.count(g_id) for g_id in g_ids)

    genes_ids = [get_gene_name(gene_id) for gene_id in species_genes_ids]
    vector = []
    for cluster_id, cluster_seqs_ids in clusters.items():
        count = count_genes_in_cluster(genes_ids, cluster_seqs_ids)
        vector.append(count)
    return vector


def run_mmseqs_pipeline(records_file: str,
                        out_directory: str,
                        species_gene_map: Dict[str, List[str]],
                        min_seq: float = 0.8,
                        cov: float = 0.8,
                        cov_mode: int = 0,
                        force_new_mmseqs: bool = False) -> (Dict[str, List[int]], List[str]):
    """Run mmseqs pipeline and obtain vectors from sequences"""
    clusters_tsv = run_mmseqs(
        records_file,
        out_directory,
        min_seq=min_seq,
        cov=cov,
        cov_mode=cov_mode,
        force_new_mmseqs=force_new_mmseqs
    )
    clusters = parse_mmseqs_file(clusters_tsv)
    vectors = {}
    for species, gene_list in species_gene_map.items():
        logging.info(f'Getting vector for: {species}')
        species_vec = get_species_clusters_vector(gene_list, clusters)
        vectors[species] = species_vec
    return vectors, list(clusters.keys())
