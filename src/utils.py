import json
import logging
import timeit
from collections import defaultdict, Counter
from functools import wraps
from pathlib import Path
from typing import List, Dict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from matplotlib import pyplot as plt


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


def plot_ontology_stats(ontology_dir: str, ontology_file: str, out_file: str, out_plot_file: str):
    with open(ontology_file) as fo:
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
    for ontology_path in Path(ontology_dir).glob('*.json'):
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
    with open(out_file, 'w') as f:
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
        plt.savefig(out_plot_file)
        plt.show()

    show_plot(points_counters)


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
        logging.info(f'Extracted {len(data)} {extract_filter} proteins for: {name}')

    logging.info(f'Loaded {len(records)} {extract_filter} proteins data for {len(species)} species')
    return records, genes_mapping


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f'[TIMING] Executing "{f.__name__}"...')
        start = timeit.default_timer()
        result = f(*args, **kwargs)
        end = timeit.default_timer()
        logging.info(f'[TIMING] Finished calculation, elapsed time = {end - start:.2f} seconds')
        return result

    return wrapper
