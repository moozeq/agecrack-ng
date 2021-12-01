import json
import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from compress_pickle import compress_pickle
from keras import Sequential
from keras.models import load_model
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from src.utils import timing


@dataclass
class ModelsConfig:
    models_reuse: bool = False
    plots_show: bool = False
    plots_annotate: bool = False
    plots_annotate_threshold: float = 0.5
    rand: int = 1
    bins: int = 0
    plots_save: bool = True


class Model:
    def __init__(self, results: dict, species_map: Dict[str, dict], class_filter: str, ontology_file: str):
        self.clusters, self.species = results['clusters'], results['species']
        with open(ontology_file) as f:
            self.ontology = json.load(f)

        def species_filter(sp: str) -> bool:
            if sp not in species_map:
                return False

            return species_map[sp]['AnAgeEntry'].species_class == class_filter if class_filter else True

        self.species = {
            sp_name: sp_data
            for sp_name, sp_data in self.species.items()
            if species_filter(sp_name)
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

        # Load dataset
        features = {
            cluster: get_column(i)
            for i, cluster in enumerate(self.clusters)
        }
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

    @classmethod
    def from_file(cls, results_filename: str, species_map: Dict[str, dict], class_filter: str, ontology_file: str):
        logging.info(f'Creating {cls.__name__} data object from: {results_filename}')
        results = cls.read_results_file(results_filename)
        return cls(results, species_map, class_filter, ontology_file)

    @classmethod
    def from_dict(cls, results: dict, species_map: Dict[str, dict], class_filter: str, ontology_file: str):
        logging.info(f'Creating {cls.__name__} data object from dict')
        return cls(results, species_map, class_filter, ontology_file)

    @staticmethod
    def read_results_file(results_filename: str) -> dict:
        with open(results_filename) as f:
            if results_filename.endswith('.gz'):
                results = compress_pickle.load(results_filename)
            else:
                results = json.load(f)
        return results

    @abstractmethod
    def train_model(self, X_train: list, y_train: list, params: dict):
        """[Method needs to be overload] Return trained model"""

    @abstractmethod
    def save_model(self, model_file: str):
        """[Method needs to be overload] Save trained model"""

    @abstractmethod
    def load_model(self, model_file: str):
        """[Method needs to be overload] Load trained model"""

    @abstractmethod
    def get_add_text(self):
        """[Method needs to be overload] Get model additional info for plot"""

    @abstractmethod
    def get_ext(self):
        """[Method needs to be overload] Get model save/load extension"""

    @abstractmethod
    def get_ontology(self, X_test, y_test) -> dict:
        """[Method needs to be overload] Get model ontology for selected features"""

    @timing
    def process(self,
                params: dict,
                out_dir: str,
                models_config: ModelsConfig) -> float:
        """Train model on data and return train score and test score"""
        model_name = type(self).__name__

        def map_type(t):
            if type(t) == str:
                return t
            elif type(t) == int:
                return str(t)
            elif type(t) == float:
                return f'{t:.1e}'
            else:
                return 'X'

        file_suffix = '_'.join(
            map_type(p)
            for p in (list(params.values()) + [models_config.rand])
        )
        model_file = f'{out_dir}/models/model_{file_suffix}_{model_name}{self.get_ext()}'
        for new_dir in ['plots', 'models', 'ontology']:
            Path(f'{out_dir}/{new_dir}').mkdir(parents=True, exist_ok=True)

        logging.info(f'Processing {model_name} data object')
        X: DataFrame = self.data[self.clusters]  # Features
        y: DataFrame = np.log(self.data['longevity'])  # Longevity in ln(years)

        # Split dataset into training set and test set
        bins_count = len(y) // 2 if not models_config.bins else models_config.bins
        bins = np.linspace(0, len(y), bins_count)
        y_binned = np.digitize(y, bins)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=models_config.rand,
                                                            stratify=y_binned)
        logging.info(f'y_train = ({min(y_train):.2f}, {max(y_train):.2f}, {mean(y_train):.2f}), '
                     f'y_test = ({min(y_test):.2f}, {max(y_test):.2f}, {mean(y_test):.2f})')

        # Create/load model
        if Path(model_file).exists() and models_config.models_reuse:
            self.load_model(model_file)
        else:
            logging.info(f'Training model on {len(y_train)} species')
            self.train_model(X_train, y_train, params)
            self.save_model(model_file)

        scores_test = self.model.score(X_test, y_test)
        logging.info(f'Model has been trained, score = {scores_test:.4f}')

        train_file = f'{out_dir}/plots/train_{file_suffix}_{model_name}.png'
        test_file = f'{out_dir}/plots/test_{file_suffix}_{model_name}.png'

        if ontology := self.get_ontology(X_test, y_test):
            coef_file = f'{out_dir}/ontology/ontology_{file_suffix}_{model_name}.json'
            with open(coef_file, 'w') as f:
                json.dump(ontology, f, indent=4)

        # self.predict_plot(X_train, y_train, f'Training data {str(params)}', train_file, show, save)
        # self.predict_plot(X_test, y_test, f'Testing data {str(params)}', test_file, show, save)
        self.predict_plot(X_train, y_train, f'Training data', train_file, models_config)
        self.predict_plot(X_test, y_test, f'Testing data', test_file, models_config)

        return scores_test

    def visualize_data(self, species_map: Dict[str, dict], seqs_filter: str, out_directory: str):
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
        plt.title(f'Clustered vertebrates {seqs_filter} proteins\' sequences')
        plt.show()
        g.savefig(f'{out_directory}/vectors_{seqs_filter}.png', dpi=600)

    def predict_plot(self, _x, _y, title: str, file: str, models_config: ModelsConfig):
        y_pred = self.model.predict(_x)
        if isinstance(_y, pd.Series):
            _y = _y.values
        if isinstance(y_pred[0], ndarray):
            y_pred = np.array([yp[0] for yp in y_pred])
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
                  f'p-value < {Model.get_pval(_y, y_pred):.2e}, '
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
        plt.clf()

    @staticmethod
    def get_r2(x_values, y_values):
        my_fitting = np.polyfit(x_values, y_values, 1, full=True)

        # Residual or Sum of Square Error (SSE)
        SSE = my_fitting[1][0]

        # Determining the Sum of Square Total (SST)
        # the squared differences between the observed dependent variable and its mean
        diff = y_values - y_values.mean()
        square_diff = diff ** 2
        SST = square_diff.sum()

        # Now getting the coefficient of determination (R2)
        R2 = 1 - SSE / SST
        return R2

    @staticmethod
    def get_pval(x_values, y_values):
        # stat, pval_res = stats.ttest_ind(x_values, y_values)
        coeff, pval = pearsonr(x_values, y_values)
        logging.info(f'coeff = {coeff:.4f}, p-value = {pval:.2e}')
        return pval

    @staticmethod
    def scatter2d(results: List[dict], scores: int = 1):
        fig = plt.figure()
        ax = fig.add_subplot()

        colors = ['red', 'green', 'blue']
        x = [r['params'][1] for r in results]
        y = [r['scores'][scores] for r in results]
        c = [colors[r['params'][2]] for r in results]
        ax.scatter(x, y, marker='o', c=c)

        ax.set_xlabel('cov')
        ax.set_ylabel('score')

        plt.show()

    @staticmethod
    def scatter3d(results: List[dict], scores: int = 1, color_by_score: bool = True):
        """Scatter3D plot for mmseq results. `scores` 0 for training scores, 1 for test scores."""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x = [r['params'][0] for r in results]
        y = [r['params'][1] for r in results]
        z = [r['scores'][scores] for r in results]
        c = [r['scores'][scores] for r in results]
        if color_by_score:
            c = [r['scores'][scores] for r in results]
            ax.scatter(x, y, z, c=c, cmap='Reds_r')
        else:
            # color by species
            colors = ['red', 'green', 'blue']
            markers = ['o', '^', 's']
            c = [colors[r['params'][2]] for r in results]
            m = [markers[r['params'][2]] for r in results]
            ax.scatter(x, y, z, c=c, marker=m, cmap='Reds_r')

        ax.set_xlabel('min_seq')
        ax.set_ylabel('cov')
        ax.set_zlabel('score')

        plt.show()

    @staticmethod
    def get_features(clusters: dict, model_features: list, threshold: float = 0.0) -> dict:
        """"""
        clusters_scores = {
            clusters[i]: feature_score
            for i, feature_score in enumerate(model_features)
            if abs(feature_score) > threshold
        }
        return dict(sorted(clusters_scores.items(), key=lambda x: x[1], reverse=True))


class ANN(Model):
    def train_model(self, X_train: list, y_train: list, params: dict):
        # create ANN model
        self.model = Sequential()

        self.model.add(layers.Conv1D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], 1, 20)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv1D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv1D(64, (3, 3), activation='relu'))
        self.model.add(layers.Dense(1, activation='relu'))

        # Fitting the ANN to the Training set
        self.model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)

    def save_model(self, model_file: str):
        self.model.save(model_file)

    def load_model(self, model_file: str):
        self.model = load_model(model_file)

    def get_add_text(self):
        return f'layers = {len(self.model.layers)}'

    def get_ext(self):
        return '.h5'

    def get_ontology(self, X_test, y_test) -> dict:
        return {}


class RF(Model):
    def train_model(self, X_train, y_train, params):
        self.model = RandomForestRegressor(**params, n_jobs=-1)

        # Train the model using the training sets y_pred=self.model.predict(X_test)
        self.model.fit(X_train, y_train)

    def save_model(self, model_file: str):
        compress_pickle.dump(self.model, model_file)

    def load_model(self, model_file: str):
        self.model = compress_pickle.load(model_file)

    def get_add_text(self):
        return f'estimators = {len(self.model.estimators_)}'

    def get_ext(self):
        return '.gz'

    def get_ontology(self, X_test, y_test) -> dict:
        return {
            cluster: {
                'coef': coef,
                'desc': self.ontology[cluster]
            }
            for cluster, coef in self.get_features(self.clusters, self.model.feature_importances_).items()
        }

        # TODO: add plots for ontology for each model
        # TODO: change coef to score
        # importances = self.model.feature_importances_
        # df = pd.DataFrame.from_dict({'importances': list(importances), 'gene': list(self.ontology)})
        # df = df[df.importances > 0.003].sort_values(by='importances')
        #
        # fig, ax = plt.subplots()
        # df.plot.bar(x='gene', y='importances', ax=ax)
        # ax.set_title("Feature importances using MDI")
        # ax.set_ylabel("Mean decrease in impurity")
        # fig.tight_layout()
        # plt.show()
        # return {}

    def visualize_tree(self):
        """Visualize random forest tree estimator"""
        if not self.model:
            logging.warning(f'No model was found in RF object to visualize')
            return

        # Extract single tree
        estimator = self.model.estimators_[0]

        from sklearn.tree import export_graphviz
        # Export as dot file
        export_graphviz(estimator, out_file='tree.dot',
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        # Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


class EN(Model):
    def train_model(self, X_train, y_train, params):
        self.model = ElasticNet(**params, max_iter=10000)

        # Train the model using the training sets
        self.model.fit(X_train, y_train)

    def save_model(self, model_file: str):
        compress_pickle.dump(self.model, model_file)

    def load_model(self, model_file: str):
        self.model = compress_pickle.load(model_file)

    def get_add_text(self):
        return f'coefs = {len(self.model.coef_)}'

    def get_ext(self):
        return '.gz'

    def get_ontology(self, X_test, y_test) -> dict:
        return {
            cluster: {
                'coef': coef,
                'desc': self.ontology[cluster]
            }
            for cluster, coef in self.get_features(self.clusters, self.model.coef_).items()
        }


class ENCV(Model):
    def train_model(self, X_train, y_train, params):
        self.model = ElasticNetCV(**params, max_iter=10000, n_jobs=-1)

        # Train the model using the training sets
        self.model.fit(X_train, y_train)

    def save_model(self, model_file: str):
        compress_pickle.dump(self.model, model_file)

    def load_model(self, model_file: str):
        self.model = compress_pickle.load(model_file)

    def get_add_text(self):
        return f'coefs = {len(self.model.coef_)}'

    def get_ext(self):
        return '.gz'

    def get_ontology(self, X_test, y_test) -> dict:
        return {
            cluster: {
                'coef': coef,
                'desc': self.ontology[cluster]
            }
            for cluster, coef in self.get_features(self.clusters, self.model.coef_).items()
        }
