import json
import logging
import math
from pathlib import Path
from statistics import mean
from typing import List

import compress_pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import absolute
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score


class RF:
    def __init__(self, results_filename: str):
        logging.info(f'Creating Random Forest data object from: {results_filename}')
        with open(results_filename) as f:
            results = json.load(f)
            self.clusters, self.species = results['clusters'], results['species']

        def get_column(index: int) -> list:
            col = [
                s['vec'][index]
                for s in self.species.values()
            ]
            return col

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

    def process(self, rf_params: dict, out_dir: str, show: bool = False, save: bool = False) -> (float, float):
        file_suffix = '_'.join(str(p) for p in rf_params.values())
        model_file = f'{out_dir}/models/model_{file_suffix}.gz'
        for new_dir in ['plots', 'models']:
            Path(f'{out_dir}/{new_dir}').mkdir(parents=True, exist_ok=True)

        logging.info(f'Processing Random Forest data object')
        X = self.data[self.clusters]  # Features
        y = np.log(self.data['longevity'])  # Longevity in ln(years)

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

        # Create a Gaussian Classifier
        if not Path(model_file).exists():
            clf = RandomForestRegressor(**rf_params, n_jobs=-1)

            # Train the model using the training sets y_pred=clf.predict(X_test)
            clf.fit(X_train, y_train)
            compress_pickle.dump(clf, model_file)
        else:
            clf = compress_pickle.load(model_file)

        scores = cross_val_score(clf, X, y,
                                 scoring='neg_mean_absolute_error',
                                 cv=KFold(n_splits=5, random_state=1, shuffle=True),
                                 n_jobs=-1)
        scores_test = sum(abs(yp - yt) for yp, yt in zip(clf.predict(X_test), y_test))
        res = mean(absolute(scores))

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

        def get_pval(x_values, y_values):
            # stat, pval_res = stats.ttest_ind(x_values, y_values)
            coeff, pval = pearsonr(x_values, y_values)
            print(f'{coeff} {pval}')
            return pval

        def pred2(_x, _y, title: str, file: str):
            y_pred = clf.predict(_x)
            plt.scatter(_y, y_pred)
            plt.ylabel('Predicted Lifespan (ln)')
            plt.xlabel('Known Lifespan (ln)')
            params_str = f'(est = {rf_params["n_estimators"]}, depth = {rf_params["max_depth"]})'
            plt.title(f'{title}, R2 = {get_r2(_y, y_pred):.2f}, p-value < {get_pval(_y, y_pred):.2e} {params_str}')

            # regression line
            coef = np.polyfit(np.array(_y), np.array(y_pred), 1)
            poly1d_fn = np.poly1d(coef)
            max_r = math.ceil(max((*_y, *y_pred)))
            sx = list(range(max_r + 1))
            plt.xlim((0.0, max_r))
            plt.ylim((0.0, max_r))
            plt.plot(sx, poly1d_fn(sx), color='red', linestyle='--')
            plt.grid()
            if save:
                plt.savefig(file)
            if show:
                plt.show()

        train_file = f'{out_dir}/plots/train_{file_suffix}.png'
        test_file = f'{out_dir}/plots/test_{file_suffix}.png'
        pred2(X_train, y_train, 'Training data', train_file)
        pred2(X_test, y_test, 'Testing data', test_file)

        return res, scores_test

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
    def scatter3d(results: List[dict], scores: int = 1):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # colors = ['red', 'green', 'blue']
        # markers = ['o', '^', 's']
        x = [r['params'][0] for r in results]
        y = [r['params'][1] for r in results]
        z = [r['scores'][scores] for r in results]
        c = [r['scores'][scores] for r in results]
        # c = [colors[r['params'][2]] for r in results]
        # m = [markers[r['params'][2]] for r in results]
        ax.scatter(x, y, z, c=c, cmap='Reds_r')

        ax.set_xlabel('min_seq')
        ax.set_ylabel('cov')
        ax.set_zlabel('score')

        plt.show()
