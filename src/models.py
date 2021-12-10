import json
import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Type, Union

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from compress_pickle import compress_pickle
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split

from src.anage import AnAgeDatabase
from src.mmseq import MmseqConfig
from src.ontology import create_ontology
from src.utils import timing, load_and_add_results, convert_ys


@dataclass
class ModelsConfig:
    models_reuse: bool = False
    plots_show: bool = False
    plots_annotate: bool = False
    plots_annotate_threshold: float = 0.5
    plots_clusters_count: int = 10
    rand: int = 1
    bins: int = 0
    stratify: bool = False
    files_additional_suffix: str = ''
    plots_save: bool = True


class Model:
    def __init__(self, results: Union[str, dict], species_map: Dict[str, dict], class_filter: str, ontology_file: str):
        if type(results) == str:
            results = Model._read_vectors_file(results)

        self.clusters, self.species = results['clusters'], results['species']
        with open(ontology_file) as f:
            self.ontology = json.load(f)

        def species_filter(sp: str) -> bool:
            if sp not in species_map:
                return False

            return species_map[sp]['AnAgeEntry'].species_class == class_filter if class_filter else True

        # filter out species which shouldn't be included in further analysis
        # e.g. not in a proper class (Mammalia only)
        self.species = {
            sp_name: sp_data
            for sp_name, sp_data in self.species.items()
            if species_filter(sp_name)
        }

        # map int index to species name
        self.index_to_species = {
            i: species_name
            for i, species_name in enumerate(self.species)
        }

        def get_column(index: int) -> list:
            col = [
                species_data['vec'][index]
                for species_name, species_data in self.species.items()
            ]
            return col

        def get_colors(sp_map: dict) -> (dict, dict):
            """Get mapping species index -> phylo color."""
            sp_idx = {
                i: sp_map[sp]['AnAgeEntry'].species_class
                for i, sp in enumerate(self.species)
            }

            phylo_classes = sorted(set(sp_idx.values()))

            sp_phylo_colors = {
                phylo: (cm.cubehelix(i / (len(phylo_classes) + 1)))
                for i, phylo in enumerate(phylo_classes)
            }

            # phylo to color index
            sp_col_map = {
                i: sp_phylo_colors[phylo]
                for i, phylo in sp_idx.items()
            }
            return sp_phylo_colors, sp_col_map

        non_zero_clusters = sum(
            sum(get_column(i)) > 0
            for i, cluster in enumerate(self.clusters)
        )

        # Load dataset
        features = {
            cluster: get_column(i)
            for i, cluster in enumerate(self.clusters)
        }

        logging.info(f'Found {non_zero_clusters} ({non_zero_clusters / len(features) * 100.0:.1f}%) non-zero clusters '
                     f'within ({len(features)}) clusters across ({len(self.species)}) species')

        target = [
            s['longevity']
            for s in self.species.values()
        ]

        self.data = pd.DataFrame({
            **features,
            'longevity': target
        })

        self.phylo_colors, self.phylo_colors_map = get_colors(species_map)

        # model will be stored in object after calling `process`
        self.model = None

    @property
    @abstractmethod
    def features(self) -> list:
        """[Property needs to be overload] Return features importance"""

    @abstractmethod
    def train_model(self, X_train: list, y_train: list, params: dict, models_config: ModelsConfig):
        """[Method needs to be overload] Return trained model"""

    @abstractmethod
    def get_add_text(self):
        """[Method needs to be overload] Get model additional info for plot"""

    @staticmethod
    @abstractmethod
    def analysis_check(records_file: str,
                       vectors_file: str,
                       species_map: Dict[str, dict],
                       class_filter: str,
                       proteins_count: str,
                       grid_params: dict,
                       ontology_file: str,
                       mmseq_config: MmseqConfig,
                       models_config: ModelsConfig,
                       anage_db: AnAgeDatabase,
                       out_directory: str):
        """[Method needs to be overload] For all params in ``grid_params`` run analysis"""

    @staticmethod
    def _read_vectors_file(results_filename: str) -> dict:
        with open(results_filename) as f:
            if results_filename.endswith('.gz'):
                results = compress_pickle.load(results_filename)
            else:
                results = json.load(f)
        return results

    @staticmethod
    def _get_pval(x_values, y_values):
        coeff, pval = pearsonr(x_values, y_values)
        return pval

    @staticmethod
    def get_features(clusters: dict, model_features: list, threshold: float = 0.0) -> dict:
        """Returns sorted dict with scores for each cluster"""
        clusters_scores = {
            clusters[i]: feature_score
            for i, feature_score in enumerate(model_features)
            if abs(feature_score) >= threshold
        }
        return dict(sorted(clusters_scores.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def run_analysis(records_file: str,
                     out_directory: str,
                     species_map: Dict[str, dict],
                     params: dict,
                     class_filter: str,
                     ontology_file: str,
                     mmseq_config: MmseqConfig,
                     models_config: ModelsConfig,
                     anage_db: AnAgeDatabase,
                     model_class: Type['Model'],
                     results: Union[str, dict]) -> tuple[dict, 'Model']:
        """
        Run full analysis of extracted proteins:
            1. Cluster sequences with provided parameters
            2. For each species create vector with counts of genes in each cluster
            3. Run ``Regressor`` to find predictor

        Attributes:
            results: dictionary with vectors data for speeding up calculation when multiple models
                created at once, or string to file with vectors otherwise
        """

        # create file with ontology if does not exists or reloading
        if not Path(ontology_file).exists() or mmseq_config.force_new_mmseqs or mmseq_config.reload_mmseqs:
            create_ontology(records_file, ontology_file, mmseq_config.clusters_file)

        # speeding up calculation when passing data as dict loaded from file before
        model = model_class(results, species_map, class_filter, ontology_file)
        score = model.process(params, out_directory, models_config)

        m_results = {
            'mmseqs_params': [mmseq_config.min_seq_id, mmseq_config.c, mmseq_config.cov_mode],
            'params': params,
            'score': score
        }
        return m_results, model

    @property
    def _model_name(self):
        return type(self).__name__

    def _save_model(self, model_file: str):
        """Save trained model to file"""
        compress_pickle.dump(self.model, model_file)

    def _load_model(self, model_file: str):
        """Load trained model from file"""
        self.model = compress_pickle.load(model_file)

    @staticmethod
    def _process_data(data):
        return np.log(data)

    @staticmethod
    def _unprocess_data(data):
        return math.e ** data

    def _save_predicted_results(self, X_train, X_test, y_train, y_test, results_file: str):
        """Save real and predicted longevity for each species"""
        # score model using test data not used in training
        scores_train = self.model.score(X_train, y_train)
        scores_test = self.model.score(X_test, y_test)

        # log and save predicted results
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        def get_results(y, yp):
            unsorted_results = {
                self.index_to_species[index]: {
                    'known': (raw_known := self._unprocess_data(known_longevity)),
                    'pred': (raw_pred := self._unprocess_data(yp[i])),
                    'diff': abs(raw_known - raw_pred)
                }
                for i, (index, known_longevity) in enumerate(y.iteritems())
            }
            # sort by difference between known and predicted longevity
            return dict(sorted(unsorted_results.items(), key=lambda item: item[1]['diff']))

        results = {
            'train': get_results(y_train, y_pred_train),
            'test': get_results(y_test, y_pred_test)
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        logging.critical(f'Model {self._model_name} (train, test): '
                         f'score = ({scores_train:.2f}, {scores_test:.2f}), '
                         f'p-value = ({Model._get_pval(y_train, y_pred_train):.2e}, {Model._get_pval(y_test, y_pred_test):.2e}), '
                         f'{self.get_add_text()}')

    def get_ontology(self) -> dict:
        """Get scores for all clusters used in model with descriptions"""
        return {
            cluster: {
                'score': score,
                'desc': self.ontology[cluster]
            }
            for cluster, score in self.get_features(self.clusters, list(self.features)).items()
        }

    @timing
    def process(self,
                params: dict,
                out_dir: str,
                models_config: ModelsConfig) -> float:
        """Train model on data and return test score"""

        def map_type(t):
            if type(t) == str:
                return t
            elif type(t) == int:
                return str(t)
            elif type(t) == float:
                return f'{t:.1e}'
            else:
                return 'X'

        # prepare file suffix based on parameters
        file_suffix = '_'.join(
            map_type(p)
            for p in (list(params.values()) + [models_config.rand])
        )
        file_suffix = f'{file_suffix}{models_config.files_additional_suffix}'

        # create directories structure
        for new_dir in ['validation', 'clusters', 'models', 'ontology', 'results', 'vectors']:
            Path(f'{out_dir}/{new_dir}').mkdir(parents=True, exist_ok=True)

        # file with model is binary compressed to .gz format
        model_file = f'{out_dir}/models/model_{file_suffix}_{self._model_name}.gz'

        # vectors of counts sequences in clusters, order is preserved
        X: DataFrame = self.data[self.clusters]
        # longevity in ln(years)
        y: DataFrame = self._process_data(self.data['longevity'])

        # log model name and parameters
        params_str = ', '.join(f'{param} = {value}' for param, value in params.items())
        logging.info(f'Processing {self._model_name} model with parameters: {params_str}')

        # split dataset into training set and test set
        # stratify is optional, but set to True by default
        if models_config.stratify:
            bins_count = len(y) // 2 if not models_config.bins else models_config.bins
            bins = np.linspace(0, len(y), bins_count)
            y_binned = np.digitize(y, bins)
        else:
            y_binned = None

        # split train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=models_config.rand,
                                                            stratify=y_binned)

        # log distribution of those sets
        logging.debug(f'Dist y_train: (min = {min(y_train):.2f}, max = {max(y_train):.2f}, mean = {mean(y_train):.2f})')
        logging.debug(f'Dist y_test: (min = {min(y_test):.2f}, max = {max(y_test):.2f}, mean = {mean(y_test):.2f})')

        # create/load model
        if Path(model_file).exists() and models_config.models_reuse:
            logging.info(f'Loading {self._model_name} model from file: "{model_file}"')
            self._load_model(model_file)
        else:
            logging.info(f'Training {self._model_name} model on ({len(y_train)}/{len(y_train + y_test)}) species, '
                         f'testing on ({len(y_test)}/{len(y_train + y_test)}) species')
            self.train_model(X_train, y_train, params, models_config)
            self._save_model(model_file)

        scores_test = self.model.score(X_test, y_test)

        # save known and predicted longevity
        results_file = f'{out_dir}/results/model_{file_suffix}_{self._model_name}.json'
        self._save_predicted_results(X_train, X_test, y_train, y_test, results_file)

        # predictor efficiency plots
        train_plot_file = f'{out_dir}/validation/train_{file_suffix}_{self._model_name}.png'
        test_plot_file = f'{out_dir}/validation/test_{file_suffix}_{self._model_name}.png'
        self._predict_plot(X_train, y_train, f'Training data', train_plot_file, models_config)
        self._predict_plot(X_test, y_test, f'Testing data', test_plot_file, models_config)

        # ontology file
        ontology_file = f'{out_dir}/ontology/ontology_{file_suffix}_{self._model_name}.json'
        with open(ontology_file, 'w') as f:
            ontology = self.get_ontology()
            json.dump(ontology, f, indent=4)
            logging.debug(f'Ontology saved, clusters count = {len(ontology)}')

        # clusters importance plot
        clusters_importance_plot_file = f'{out_dir}/clusters/clusters_{file_suffix}_{self._model_name}.png'
        self._clusters_importance_plot(clusters_importance_plot_file, models_config)
        logging.debug(f'Clusters importance plot saved')

        # plot best vectors heatmap
        vectors_plot_file = f'{out_dir}/vectors/vectors_{file_suffix}_{self._model_name}.png'
        self._best_clusters_vectors_plot(ontology, models_config, vectors_plot_file)
        logging.debug(f'Vectors heatmap saved, clusters on the plot up to = {models_config.plots_clusters_count}')

        return scores_test

    def _best_clusters_vectors_plot(self, ontology: dict, models_config: ModelsConfig, vectors_plot_file: str):
        sorted_best_clusters = list(ontology)[:models_config.plots_clusters_count]
        sorted_clusters_indexes = [self.clusters.index(cls) for cls in sorted_best_clusters]
        sorted_species_by_longevity = sorted(self.species, key=lambda sp: self.species[sp]['longevity'])
        species_vecs = [
            [
                self.species[sp]['vec'][cls_index]
                for cls_index in sorted_clusters_indexes
            ]
            for sp in sorted_species_by_longevity
        ]
        species_labels = [
            f'{sp} ({self.species[sp]["longevity"]} yrs)'
            for sp in sorted_species_by_longevity
        ]
        sorted_df = pd.DataFrame(species_vecs, columns=sorted_best_clusters, index=sorted_species_by_longevity)

        import seaborn as sns
        sns.set(rc={'figure.figsize': (16, 8)})
        g = sns.clustermap(sorted_df, col_cluster=False, row_cluster=False, yticklabels=True, xticklabels=True)

        g.ax_heatmap.set_yticklabels(species_labels, fontsize=(35 / np.sqrt(len(species_labels))))
        g.ax_heatmap.set_xticklabels(sorted_best_clusters)
        g.ax_heatmap.tick_params(bottom=False)
        g.ax_heatmap.set_title(f'Most important clusters counts heatmap, N = {models_config.plots_clusters_count}')

        if models_config.plots_save:
            g.savefig(vectors_plot_file, dpi=600)
        if models_config.plots_show:
            plt.show()
        plt.close('all')
        plt.cla()
        plt.clf()

    def visualize_vectors(self, species_map: Dict[str, dict], seqs_filter: str, out_directory: str):
        """Visualize species vectors.

        Each species has its own vector of counts for proteins in clusters.
        We can visualize those vectors and sort them with phylo and ascending
        longevity obtaining a matrix.
        """
        sorted_df = self.data

        def col_score(column):
            all_hits = sum(field for field in column if field)
            cluster_hits = sum(True for field in column if field)
            return cluster_hits / len(column), all_hits

        longevity = sorted_df['longevity']

        columns_to_drop = []
        columns_density = []
        for i, col in enumerate(sorted_df):
            data = sorted_df[col]
            score, hits = col_score(data)
            columns_density.append((col, i, hits, score))
            if score < 0.01:
                columns_to_drop.append(col)

        columns_density.sort(key=lambda x: x[3], reverse=True)
        sorted_df = sorted_df[[col[0] for col in columns_density]]

        sorted_df.drop(columns_to_drop, axis='columns', inplace=True)

        # remove longevity and genes columns
        columns_to_drop = ['longevity']
        sorted_df.drop(columns_to_drop, axis='columns', inplace=True)

        sp_idx_map = {
            i: species_map[sp]['AnAgeEntry'].species_class
            for i, sp in enumerate(self.species)
        }

        sp_idx_map = [
            sp_idx_map[idx]
            for idx in sorted_df.index
        ]

        sp_classes = sorted(set(sp_idx_map))

        sp_phylo_colors = {
            phylo: i
            for i, phylo in enumerate(sp_classes)
        }

        import seaborn as sns
        sns.set(rc={'figure.figsize': (16, 8)})

        sorted_df.insert(0, 'phylo', sp_idx_map)

        network_pal = sns.cubehelix_palette(len(sp_classes),
                                            light=.9, dark=.1, reverse=True,
                                            start=1, rot=-2)
        network_lut = dict(zip(sp_classes, network_pal))

        sorted_df.insert(0, 'longevity', [l for i, l in longevity.items()])
        sorted_df.sort_values(['phylo', 'longevity'], inplace=True)

        row_colors = sorted_df.phylo.map(network_lut)

        columns_to_drop = ['phylo', 'longevity']
        sorted_df.drop(columns_to_drop, axis='columns', inplace=True)

        g = sns.clustermap(sorted_df, row_cluster=False, row_colors=row_colors, yticklabels=True, xticklabels=True)
        for label in sp_phylo_colors:
            g.ax_heatmap.bar(0, 0, color=network_lut[label],
                             label=label, linewidth=0)
        g.ax_heatmap.legend(ncol=3, bbox_to_anchor=(1.0, 1.25))

        longevity_labels = [
            str(int(longevity[int(txt._text)]))
            for txt in g.ax_heatmap.get_yticklabels()
        ]
        g.ax_heatmap.set_yticklabels(longevity_labels, fontsize=3)
        g.ax_heatmap.set_xticklabels([])
        g.ax_heatmap.tick_params(bottom=False)
        g.ax_col_dendrogram.set_visible(False)
        g.ax_heatmap.set_title(f'Clustered vertebrates {seqs_filter} proteins sequences')
        g.savefig(f'{out_directory}/vectors.png', dpi=600)
        plt.show()

    def _clusters_importance_plot(self, file: str, models_config: ModelsConfig):
        df = pd.DataFrame.from_dict({'scores': list(self.features), 'cluster': list(self.clusters)})
        df = df.sort_values(by='scores', ascending=False).head(models_config.plots_clusters_count)
        fig, ax = plt.subplots()
        df.plot.bar(x='cluster', y='scores', ax=ax)
        fig.tight_layout()
        plt.grid()
        if models_config.plots_save:
            plt.savefig(file)
        if models_config.plots_show:
            plt.show()
        ax.clear()
        plt.close('all')
        plt.cla()
        plt.clf()

    def _predict_plot(self, _x, _y, title: str, file: str, models_config: ModelsConfig):
        """Make prediction scatter plot on train/test data with R2 and p-value in the title"""
        y_pred = self.model.predict(_x)
        _y, y_pred = convert_ys(_y, y_pred)
        phylo_colors = [
            self.phylo_colors_map[idx]
            for idx in _x.index
        ]
        markers = [plt.Line2D([0, 0], [0, 0], color=c, marker='o', linestyle='') for c in self.phylo_colors.values()]
        plt.legend(markers, self.phylo_colors, numpoints=1)

        if models_config.plots_annotate:
            # map index from DataFrame to species name
            sp_map = {
                index: sp
                for index, sp in enumerate(self.species)
            }
            # annotate proper points on scatter plot
            for i, idx in enumerate(_x.index):
                if abs(_y[i] - y_pred[i]) > models_config.plots_annotate_threshold:
                    plt.annotate(sp_map[idx], (_y[i], y_pred[i]))

        plt.scatter(_y, y_pred, c=phylo_colors)
        plt.ylabel('Predicted Lifespan (ln)')
        plt.xlabel('Known Lifespan (ln)')
        plt.title(f'{title}, R2 = {self.model.score(_x, _y):.2f}, '
                  f'p-value < {Model._get_pval(_y, y_pred):.2e}, '
                  f'{self.get_add_text()}')

        # regression line
        coef = np.polyfit(np.array(_y), np.array(y_pred), 1)
        poly1d_fn = np.poly1d(coef)
        max_r = math.ceil(max((*_y, *y_pred)))
        sx = list(range(max_r + 1))
        plt.xlim((0.0, max_r))
        plt.ylim((0.0, max_r))
        plt.plot(sx, poly1d_fn(sx), color='red', linestyle='--')
        plt.plot(sx, sx, color='blue', linestyle='--')
        plt.grid()
        if models_config.plots_save:
            plt.savefig(file)
        if models_config.plots_show:
            plt.show()
        plt.close('all')
        plt.cla()
        plt.clf()


class RF(Model):
    def train_model(self, X_train, y_train, params, models_config: ModelsConfig):
        self.model = RandomForestRegressor(**params, n_jobs=-1, random_state=models_config.rand)

        # Train the model using the training sets y_pred=self.model.predict(X_test)
        self.model.fit(X_train, y_train)

    def get_add_text(self):
        return f'estimators = {len(self.model.estimators_)}'

    @property
    def features(self):
        return self.model.feature_importances_

    def visualize_tree(self, estimator_index: int):
        """Visualize random forest tree estimator"""
        if not self.model:
            logging.warning(f'No model was found in RF object to visualize')
            return

        # Extract single tree
        estimator = self.model.estimators_[estimator_index]

        from sklearn.tree import export_graphviz
        # Export as dot file
        export_graphviz(estimator, out_file='tree.dot',
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        # Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    @staticmethod
    def analysis_check(records_file: str,
                       vectors_file: str,
                       species_map: Dict[str, dict],
                       class_filter: str,
                       proteins_count: str,
                       grid_params: dict,
                       ontology_file: str,
                       mmseq_config: MmseqConfig,
                       models_config: ModelsConfig,
                       anage_db: AnAgeDatabase,
                       out_directory: str):
        """Do Random Forest Regressor analysis for multiple parameters combinations."""
        current_results = []

        # read model from file to get it into memory, speeding up multiple calls
        results_dict = Model._read_vectors_file(vectors_file) if Path(vectors_file).exists() else {}

        # wrap params in lists if they are not
        grid_params = {
            param: [grid_params[param]] if type(grid_params[param]) != list else grid_params[param]
            for param in grid_params
        }

        for estimators in grid_params['n_estimators']:
            for depth in grid_params['max_depth']:
                rf_params = {
                    'n_estimators': estimators,
                    'max_depth': depth
                }
                results, rf = Model.run_analysis(records_file,
                                                 out_directory,
                                                 species_map,
                                                 rf_params,
                                                 class_filter,
                                                 ontology_file,
                                                 mmseq_config,
                                                 models_config,
                                                 anage_db,
                                                 RF,
                                                 results_dict)

                if proteins_count:
                    logging.info(f'Number of sequences used in analysis: {proteins_count}')
                current_results.append({
                    'model': 'RandomForest',
                    'model_params': rf_params,
                    'results': results
                })

        load_and_add_results(f'{out_directory}/check.json', current_results)


class EN(Model):
    def train_model(self, X_train, y_train, params, models_config: ModelsConfig):
        self.model = ElasticNet(**params, max_iter=10000, random_state=models_config.rand)

        # Train the model using the training sets
        self.model.fit(X_train, y_train)

    def get_add_text(self):
        # show only coefs != 0.0
        return f'coefs = {np.count_nonzero(self.model.coef_)}'

    @property
    def features(self):
        return self.model.coef_

    @staticmethod
    def analysis_check(records_file: str,
                       vectors_file: str,
                       species_map: Dict[str, dict],
                       class_filter: str,
                       proteins_count: str,
                       grid_params: dict,
                       ontology_file: str,
                       mmseq_config: MmseqConfig,
                       models_config: ModelsConfig,
                       anage_db: AnAgeDatabase,
                       out_directory: str):
        """Do Elastic Net Regressor analysis for multiple parameters combinations."""
        current_results = []

        results_dict = Model._read_vectors_file(vectors_file) if Path(vectors_file).exists() else {}
        for alpha in grid_params['alpha']:
            for l1_ratio in grid_params['l1_ratio']:
                params = {
                    'alpha': alpha,
                    'l1_ratio': l1_ratio
                }
                results, en = Model.run_analysis(records_file,
                                                 out_directory,
                                                 species_map,
                                                 params,
                                                 class_filter,
                                                 ontology_file,
                                                 mmseq_config,
                                                 models_config,
                                                 anage_db,
                                                 EN,
                                                 results_dict)

                if proteins_count:
                    logging.info(f'Number of sequences used in analysis: {proteins_count}')
                current_results.append({
                    'model': 'ElasticNet',
                    'model_params': params,
                    'results': results
                })

        load_and_add_results(f'{out_directory}/check.json', current_results)


class ENCV(Model):
    def train_model(self, X_train, y_train, params, models_config: ModelsConfig):
        self.model = ElasticNetCV(**params, max_iter=10000, n_jobs=-1, random_state=models_config.rand)

        # Train the model using the training sets
        self.model.fit(X_train, y_train)

    def get_add_text(self):
        # show only coefs != 0.0
        return f'coefs = {np.count_nonzero(self.model.coef_)}'

    @property
    def features(self):
        return self.model.coef_

    @staticmethod
    def analysis_check(records_file: str,
                       vectors_file: str,
                       species_map: Dict[str, dict],
                       class_filter: str,
                       proteins_count: str,
                       grid_params: dict,
                       ontology_file: str,
                       mmseq_config: MmseqConfig,
                       models_config: ModelsConfig,
                       anage_db: AnAgeDatabase,
                       out_directory: str):
        """Do Elastic Net Regressor with cross validation analysis"""
        current_results = []

        results_dict = Model._read_vectors_file(vectors_file) if Path(vectors_file).exists() else {}
        params = grid_params
        results, encv = Model.run_analysis(records_file,
                                           out_directory,
                                           species_map,
                                           params,
                                           class_filter,
                                           ontology_file,
                                           mmseq_config,
                                           models_config,
                                           anage_db,
                                           ENCV,
                                           results_dict)

        if proteins_count:
            logging.info(f'Number of sequences used in analysis: {proteins_count}')
        current_results.append({
            'model': 'ElasticNetCV',
            'model_params': params,
            'results': results
        })

        load_and_add_results(f'{out_directory}/check.json', current_results)
