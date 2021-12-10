import csv
import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Type

import numpy as np
from matplotlib import pyplot as plt

from src.utils import save_vectors_file


@dataclass
class MmseqConfig:
    """Mapping mmseqs2 options to Python object."""

    # file for clusters
    clusters_file: str

    # config for running ``mmseqs``
    reload_mmseqs: bool = False  # reload ``mmseqs`` files, when trying different thresholds
    force_new_mmseqs: bool = False  # force running ``mmseqs`` even if results files exist
    cluster_count_threshold: int = 0  # threshold for strength of cluster under which they are filtered out
    vectors_mode: str = 'count'  # vectors mode, indicating if sequences in clusters should be counted or boolean

    # directly passed to ``mmseqs`` (default values are estimated for this project)
    min_seq_id: float = 0.8  # List matches above this sequence identity (for clustering)
    c: float = 0.8  # List matches above this fraction of aligned (covered) residues
    cov_mode: int = 0  # 0: coverage of query and target; 1: coverage of target; 2: coverage of query


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


def _get_species_clusters_vector(species_genes_ids: List[str], clusters: dict, mmseq_config: MmseqConfig) -> List[int]:
    """For species filtered proteome and clusters from mmseqs, return vector with genes counts."""

    def get_gene_name(rec_id: str) -> str:
        return rec_id.split('|')[1]

    def count_genes_in_cluster(g_ids: set, clus: set) -> int:
        return len(clus.intersection(g_ids))

    def check_if_genes_in_cluster(g_ids: set, clus: set) -> int:
        return 1 if clus.intersection(g_ids) else 0

    genes_ids = set(get_gene_name(gene_id) for gene_id in species_genes_ids)
    vector = []
    for cluster_id, cluster_seqs_ids in clusters.items():
        if mmseq_config.vectors_mode == 'count':
            count = count_genes_in_cluster(genes_ids, cluster_seqs_ids)
        else:
            count = check_if_genes_in_cluster(genes_ids, cluster_seqs_ids)
        vector.append(count)
    return vector


def mmseq_scatter2d(results: List[dict]):
    fig = plt.figure()
    ax = fig.add_subplot()

    colors = ['red', 'green', 'blue']
    x = [r['mmseqs_params'][1] for r in results]
    y = [r['score'] for r in results]
    c = [colors[r['mmseqs_params'][2]] for r in results]
    ax.scatter(x, y, marker='o', c=c)

    ax.set_xlabel('cov')
    ax.set_ylabel('score')

    plt.show()


def mmseq_scatter3d(results: List[dict], out_file: str, color_by_score: bool = False):
    """Scatter3D plot for mmseq results. `scores` 0 for training scores, 1 for test scores."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = [r['mmseqs_params'][0] for r in results]
    y = [r['mmseqs_params'][1] for r in results]
    z = [r['score'] for r in results]
    if color_by_score:
        c = [r['score'] for r in results]
        ax.scatter(x, y, z, c=c, cmap='Reds')
    else:
        # color by cov mode
        colors = ['red', 'green', 'blue']
        c = [colors[r['mmseqs_params'][2]] for r in results]
        ax.scatter(x, y, z, c=c, cmap='Reds')

    ax.set_xlabel('min_seq')
    ax.set_ylabel('cov')
    ax.set_zlabel('score')

    plt.savefig(out_file)
    plt.show()


def mmseq_check(records_file: str,
                vectors_file: str,
                species_map: Dict[str, dict],
                out_directory: str,
                ontology_file: str,
                class_filter: str,
                params: dict,
                model_class: Type['Model'],
                anage_db: 'AnAgeDatabase',
                mmseq_config: MmseqConfig,
                models_config: 'ModelsConfig',
                min_param: float = 0.1,
                max_param: float = 0.9,
                step: float = 0.1):
    """Cluster sequences using multiple parameters combinations."""
    from src.models import Model

    # force new mmseq everytime parameters change
    mmseq_config.force_new_mmseqs = True
    current_results = []

    if not Path(p := f'{out_directory}/check_mmseqs.json').exists():
        # coverage mode is an int: 0, 1 or 2
        for cov_mode in range(3):
            mmseq_config.cov_mode = cov_mode
            for min_seq_id in np.arange(min_param, max_param, step):
                mmseq_config.min_seq_id = min_seq_id
                for c in np.arange(min_param, max_param, step):
                    mmseq_config.c = c
                    models_config.files_additional_suffix = f'_{cov_mode}_{min_seq_id}_{c}'

                    vectors, clusters = run_mmseqs_pipeline(
                        records_file,
                        species_map,
                        mmseq_config,
                        out_directory
                    )
                    save_vectors_file(vectors, clusters, anage_db, vectors_file)
                    logging.info(f'Clustering and vectorization done, '
                                 f'created ({len(vectors)}) species vectors, each with ({len(clusters)}) in length')

                    r, m = Model.run_analysis(records_file,
                                              out_directory,
                                              species_map, params, class_filter,
                                              ontology_file, mmseq_config,
                                              models_config, anage_db, model_class, vectors_file)
                    current_results.append(r)
            # update file after each cov mode
            with open(p, 'w') as f:
                json.dump(current_results, f, indent=4)

    with open(p) as f:
        all_results = json.load(f)

    mmseq_scatter3d(all_results, f'{out_directory}/mmseqs.png')


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

    logging.info(f'Number of clusters loaded = {len(clusters)}')

    species_gene_map = {
        sp: set(species_map[sp]['genes'])
        for sp in species_map
    }

    for species, gene_list in species_gene_map.items():
        species_vec = _get_species_clusters_vector(gene_list, clusters, mmseqs_config)
        vectors[species] = species_vec
        nonzero_clusters = sum(1 for i in species_vec if i)
        logging.info(f'Got vector for: {species:28}, '
                     f'nonzero clusters = {nonzero_clusters:6} ({nonzero_clusters / len(species_vec) * 100.0:.1f}%), '
                     f'vector sum = {sum(species_vec)}')

    return vectors, list(clusters.keys())
