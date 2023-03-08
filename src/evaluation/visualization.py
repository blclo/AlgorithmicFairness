from typing import Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FairnessVisualizer():

    def __init__(self, df: pd.DataFrame, biasAttribute: str):

        # Define inputs
        self.df = df
        self.biasAttribute = biasAttribute

        # Define sensitive attribute for investigation
        self.sens_attr = df.iloc[:, [biasAttribute in column for column in df.columns]].columns
        # Revert one-of-K encoding for coloring
        self.df['color_by'] = df[self.sens_attr].idxmax(1).apply(lambda x: x.split("_")[-1])


    def plot_confusion_matrices(self, nrows: int, ncols: int, figsize: Tuple[int, int], normalize: bool = False):

        accs = {}
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, sharey=True
        )

        for class_idx, col in enumerate(self.sens_attr):
            df_ = self.df.query(f"{col} == 1")
            max_val = self.df['color_by'].value_counts().max()
            
            ax = axes.flatten()[class_idx]
            cm = confusion_matrix(df_['labels'], df_['preds'])
            # Store accuracy of sensitive group
            accs[col] = np.diag(cm).sum() / np.sum(cm)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                max_val=1
            
            sns.heatmap(
                cm,
                annot=True,
                fmt="d" if not normalize else ".3f",
                cmap='Blues',
                ax=ax,
                vmin=0,
                vmax=max_val,
                cbar=False # col == self.sens_attr[-1],
            )

            # Set ticks
            ax.set_xticks([0.5, 1.5], ['Prediction = No', 'Prediction = Yes'])
            ax.set_yticks([0.5, 1.5], ['Ground truth = No', 'Ground truth = Yes'])

            # Set title
            ax.set_title(f"{col} (N={len(df_)})", fontsize=10)
            
            im = ax.imshow(cm, cmap='Blues')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        plt.colorbar(im, cax=cbar_ax, cmap='Blues')
        # Show figure
        #fig.suptitle("Independence analysis", fontsize=15)
        return fig, axes, accs
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (6, 6)):
        
        fig = plt.figure(figsize=figsize)
        for i, col in enumerate(self.sens_attr):
            df_ = self.df.query(f"{col} == 1") # Call each category within the sensitive attribute of interest

            # FPR and TPR
            fpr, tpr, threshold = roc_curve(df_['labels'], df_['pred_probs'])            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=f"C{i}", label=f"AUC = {auc(fpr, tpr):.3f} - {col.split('_')[-1]}")

        plt.plot([0, 1], [0, 1], color='gray', ls='--')

        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.legend()
        fig.suptitle("Separation analysis", fontsize=15)
        return fig
    
    def plot_calibration_curves(self, nrows: int, ncols: int, figsize: Tuple[int, int]):

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for class_idx, col in enumerate(self.sens_attr):
            ax = axes.flatten()[class_idx]
            df_ = self.df.query(f"{col} == 1")

            p_true_mean, p_true_se, p_pred = self.compute_calibration_curve(df_['labels'], df_['pred_probs'])
            ax.errorbar(p_pred, p_true_mean, p_true_se, label=col, color=f"C{class_idx}", marker='o', markersize=8)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
            ax.legend(loc='upper left')

        if self.biasAttribute == 'V4_area_origin':
            axes.flatten()[-1].axis('off')

            axes[1,0].set_xlabel('Predicted probability')
            axes[1,1].set_xlabel('Predicted probability')
            axes[0,0].set_ylabel('True probability')
            axes[1,0].set_ylabel('True probability')

        elif self.biasAttribute == 'V1_sex':
            axes[0].set_xlabel('Predicted probability')
            axes[1].set_xlabel('Predicted probability')
            axes[0].set_ylabel('True probability')

        fig.subplots_adjust(wspace=0.1)
        fig.suptitle('Sufficiency analysis', fontsize=15)
        return fig, axes
    
    def compute_calibration_curve(self, targets, probs, num_bins=10):
        # From course 02477 - Bayesian Machine Learning (Spring 2023)
        bins = np.linspace(0, 1, num_bins+1)

        p_true_mean, p_true_se, p_pred = [], [], []

        for i in range(num_bins):
            bin_start, bin_end = bins[i], bins[i+1]
            bin_center = 0.5*(bin_start + bin_end)
            
            bin_idx = np.logical_and(bin_start <= probs, probs < bin_end)
            num_points_in_bin = np.sum(bin_idx)
            
            if len(targets[bin_idx]) == 0:
                continue

            p_pred.append(bin_center)
            p_est = np.mean(targets[bin_idx])
            p_true_mean.append(p_est)
            p_true_se.append(np.sqrt(p_est*(1-p_est)/num_points_in_bin))
            
        return np.array(p_true_mean), np.array(p_true_se), np.array(p_pred)
    
    def plot_latent_representations(self, test_repr: np.ndarray, train_repr: Optional[np.ndarray] = None, reduction_method: str = 'PCA', perplexity: Optional[float] = None, figsize: Tuple[int, int] = (4, 4)):
        
        # Define reduction method
        method = PCA(n_components=2) if reduction_method == 'PCA' else TSNE(n_components=2, perplexity=perplexity)
        
        # Run reduction
        if reduction_method == 'PCA':
            model = method.fit(train_repr)
            res = model.transform(test_repr)
        else:
            res = method.fit_transform(test_repr)

        self.df[f'{reduction_method}1'], self.df[f'{reduction_method}2'] = res[:, 0], res[:, 1]

        self.df['color_by'] = self.df[self.sens_attr].idxmax(1).apply(lambda x: x.split("_")[-1])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i, (name, sub_df) in enumerate(self.df.groupby('color_by')):
            sub_df.plot(
                x=f'{reduction_method}1', 
                y=f'{reduction_method}2',
                kind='scatter',
                ax=ax, color=f"C{i}", 
                label=name, 
                alpha=0.5
            )
        
        fig.suptitle(f'Latent space analysis ({reduction_method})')
        return fig, ax