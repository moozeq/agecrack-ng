import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO
from matplotlib import pyplot as plt


@dataclass
class OntologyConfig:
    ontology_dir: str
    ontology_file: str
    out_file: str
    out_plot_file: str
    N: int = 10
    plot_save: bool = True
    plot_show: bool = False


def map_ids_to_descs(fasta_file: str) -> dict:
    def get_gene_name(rec_id: str) -> str:
        return rec_id.split('|')[1]

    def get_gene_desc(rec_desc: str) -> str:
        desc_list = rec_desc.split('OS=')[0].split()[1:]
        return ' '.join(desc_list)

    return {
        get_gene_name(record.id): get_gene_desc(record.description)
        for record in SeqIO.parse(fasta_file, 'fasta')
    }


def map_clusters_to_descs(cluster_file: str, genes_to_descs: dict) -> dict:
    with open(cluster_file) as f:
        clusters = json.load(f)
        return {
            cluster: list(set([
                genes_to_descs[seq]
                for seq in seqs
                if seq in genes_to_descs
            ]))
            for cluster, seqs in clusters.items()
        }


def map_clusters_to_descs_with_counts(cluster_file: str, genes_to_descs: dict) -> dict:
    with open(cluster_file) as f:
        clusters = json.load(f)
        return {
            cluster: dict(Counter([
                genes_to_descs[seq]
                for seq in seqs
                if seq in genes_to_descs
            ]).most_common())
            for cluster, seqs in clusters.items()
        }


def ontology_stats(ontology_config: OntologyConfig):
    """Plot frequencies for the most important clusters on first 10 places"""
    with open(ontology_config.ontology_file) as fo:
        onto_dict = json.load(fo)

    def freq_counter(o: list):
        c = Counter(o)
        freq = {clus: float(f'{(count / len(o)) * 100.0:.2f}') for clus, count in c.items()}
        freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
        return freq

    def get_onto(o: list):
        freq = freq_counter(o)
        ontos = {clus: onto_dict[clus] for clus in freq}
        return {'freqs': freq, 'ontos': ontos}

    points = defaultdict(list)
    for ontology_path in Path(ontology_config.ontology_dir).glob('*.json'):
        with open(ontology_path) as f:
            ontology = json.load(f)
            for i, cluster in enumerate(ontology):
                points[i].append(cluster)
                if i == 9:
                    break
    points_counters = {
        i: get_onto(onto)
        for i, onto in points.items()
    }
    with open(ontology_config.out_file, 'w') as f:
        json.dump(points_counters, f, indent=4)

    def get_colors_map(pts: dict):
        clusters = defaultdict(lambda: 0)
        for place, data in pts.items():
            for clus, freq in data['freqs'].items():
                clusters[clus] += freq
        clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: item[1], reverse=True)}
        clusters_len = len(clusters)

        import matplotlib.cm as cm
        clusters = {
            c: cm.gist_ncar(i / clusters_len)
            for i, c in enumerate(clusters)
        }
        return clusters

    def show_plot(pts: dict):
        c_map = get_colors_map(pts)
        fig, ax = plt.subplots(figsize=(15, 8))
        for place, data in pts.items():
            prev = 0.0
            for clus, ratio in data['freqs'].items():
                ax.bar([place], [ratio], label=clus, color=c_map[clus], bottom=prev)
                prev += ratio
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color=col, label=c)
            for c, col in c_map.items()
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc="upper left")
        if ontology_config.plot_save:
            plt.savefig(ontology_config.out_plot_file)
        if ontology_config.plot_show:
            plt.show()
        plt.cla()
        plt.clf()

    show_plot(points_counters)


def ontology_scores(ontology_config: OntologyConfig):
    """Plot frequencies for the most important clusters on first N places"""
    clusters = defaultdict(float)
    for ontology_path in Path(ontology_config.ontology_dir).glob('*.json'):
        with open(ontology_path) as f:
            ontology = json.load(f)
            for cluster in ontology:
                clusters[cluster] += ontology[cluster]['score']
    clusters = dict(sorted(clusters.items(), key=lambda item: item[1], reverse=True))

    high_clusters = {
        cls: clusters[cls]
        for cls in list(clusters)[:10]
    }

    with open(ontology_config.ontology_file) as fo:
        onto_dict = json.load(fo)
    with open(ontology_config.out_file, 'w') as f:
        clusters_with_scores = {
            cls: {
                'score': clusters[cls],
                'desc': onto_dict[cls]
            }
            for cls in clusters
        }
        json.dump(clusters_with_scores, f, indent=4)

    plt.bar(list(high_clusters), list(high_clusters.values()))
    plt.xticks(rotation=45)
    plt.ylabel(f'Absolute cluster score')
    plt.title(f'Scores from the {ontology_config.N}/{len(clusters)} most important clusters')
    plt.grid()
    plt.tight_layout()
    if ontology_config.plot_save:
        plt.savefig(ontology_config.out_plot_file)
    if ontology_config.plot_show:
        plt.show()
    plt.cla()
    plt.clf()
