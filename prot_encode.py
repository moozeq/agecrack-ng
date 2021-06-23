import logging
from pathlib import Path
from typing import List, Tuple

import esm
import torch
from compress_pickle import compress_pickle

from utils import get_name_from_filename, load_records, normalize_species

# Load ESM-1b model
MODEL, ALPHABET = esm.pretrained.esm1b_t33_650M_UR50S()
BATCH_CONVERTER = ALPHABET.get_batch_converter()
MAX_SEQ_LEN = 1022


class ESM:
    @staticmethod
    def process_files(species_fastas: List[str]):
        for fn in species_fastas:
            species, data = ESM.get_formatted_data(fn)

            logging.info(f'Processing data for: {species}')
            for i in range(min(12, len(data))):
                ESM.process_single(species, data[i])

    @staticmethod
    def get_formatted_data(fasta_file: str) -> (str, List[Tuple[str, str]]):
        seqs = [(rec.id, str(rec.seq[:MAX_SEQ_LEN])) for rec in load_records(fasta_file)]
        species = get_name_from_filename(fasta_file)
        species = normalize_species(species)
        return species, seqs

    @staticmethod
    def process_single(species: str, data: Tuple[str, str]):
        sp_path_ten = f'data/tensors/{species}'
        sp_path_plot = f'data/plots/{species}'
        fig_file = f'{sp_path_plot}_{data[0]}.png'
        tensor_file = f'{sp_path_ten}_{data[0]}.gz'

        if Path(fig_file).exists() and Path(tensor_file).exists():
            return

        batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER([data])

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = MODEL(batch_tokens, repr_layers=[33], return_contacts=True)

        # Look at the unsupervised self-attention map contact predictions
        import matplotlib.pyplot as plt
        for (seq_id, seq), attention_contacts in zip([data], results["contacts"]):
            plt.matshow(mat := attention_contacts[: len(seq), : len(seq)])
            plt.title(seq_id)
            plt.savefig(fig_file)
            plt.clf()
            with open(tensor_file, 'wb') as f:
                compress_pickle.dump(mat.numpy().tolist(), f, compression='gzip')

        logging.info(f'[DONE] processing sequence {data[0]}')

    @staticmethod
    def process_single_parallel(species: str, data: List[Tuple[str, str]]):
        batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER(data)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = MODEL(batch_tokens, repr_layers=[33], return_contacts=True)

        # token_representations = results["representations"][33]
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        # output = {}
        # for i, (seq_id, seq) in enumerate(data):
        #     output[seq_id] = token_representations[i, 1: len(seq) + 1].mean(0)

        sp_path_ten = f'data/tensors/{species}'
        sp_path_plot = f'data/plots/{species}'

        # Look at the unsupervised self-attention map contact predictions
        import matplotlib.pyplot as plt
        for (seq_id, seq), attention_contacts in zip(data, results["contacts"]):
            plt.matshow(mat := attention_contacts[: len(seq), : len(seq)])
            plt.title(seq_id)
            plt.savefig(f'{sp_path_plot}_{seq_id}.png')
            # plt.show()
            # torch.save(mat, f'{sp_path_ten}_{seq_id}.pt')
            with open(f'{sp_path_ten}_{seq_id}.gz', 'wb') as f:
                compress_pickle.dump(mat.numpy().tolist(), f, compression='gzip')

        logging.info(f'[DONE] processing batch of {len(data)} sequences')
