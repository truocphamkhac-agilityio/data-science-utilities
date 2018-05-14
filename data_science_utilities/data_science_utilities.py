# -*- coding: utf-8 -*-

"""Main module."""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm

from sklearn.model_selection import learning_curve


def missing_data_stats(df):
    """Check remaining missing value and print out if any.

    Arguments:
        df (pd.DataFrame): The dataframe need to check

    Returns:
        The missing dataframe

    Usage:
        from data_science_utilities import data_science_utilities
        # make statistic
        missing_data = data_science_utilities.missing_data_stats(df)
        # display missing data
        missing_data
    """
    nan_df = (df.isnull().sum() / len(df)) * 100
    nan_df = nan_df.drop(nan_df[nan_df == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio': nan_df})

    return missing_data


def read_csv_files(*csv_files, **read_csv_kwargs):
    """Read csv files to get dataframe.

    Arguments:
        csv_files (str): path to csv files params.
        read_csv_kwargs: keyword-only arguments for read_csv pandas

    Returns:
        A tuple dataframes

    Usage:
        from src.utils.utils import read_csv_files
    """
    df_list = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, **read_csv_kwargs)
        df_list.append(df)

    return tuple(df_list)


def plot_dist_norm(dist, title):
    """Plotting normal distribution.

    Args:
        dist (pd.DataFrame): The distribution dataframe
        title (str): The title of chart

    Usage:
        from src.visualization.visualize import plot_dist_norm
    """
    sns.distplot(dist, fit=norm)
    (mu, sigma) = norm.fit(dist)
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.figure()
    stats.probplot(dist, plot=plt)
    plt.show()


def plot_corelation_matrix(data):
    """
    Plotting the co-relation matrix on the dataset
    using the numeric columns only.
    """
    corr = data.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 22))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0.0,
        vmax=1, square=True, linewidths=.5, ax=ax
    )

    return corr


def plot_top_corelation_matrix(data, target, k=10, cmap='YlGnBu'):
    """
    Plotting top k features with highest co-relation matrix.
    """
    corr = data.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()
    cols = corr.nlargest(k, target)

    # Drop columns that not in the top co-relation.
    excludes = corr.index.map(lambda x: x if x not in cols.index else None)
    excludes = excludes.dropna()

    cols_index = cols[target].index
    cols.drop(excludes, axis=1, inplace=True)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(cols, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 11))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        cols, mask=mask, cmap=cmap, center=0.0,
        vmax=1, square=True, linewidths=.5, ax=ax,
        yticklabels=cols_index.values, xticklabels=cols_index.values,
    )

    return cols_index


def plot_corelation_scatter(data, column_name, target):
    """
    Plotting scatter for corelation column.
    """
    if column_name == target:
        return

    fig, ax = plt.subplots()
    ax.scatter(x=data[column_name], y=data[target])
    plt.xlabel(column_name, fontsize=13)
    plt.ylabel(target, fontsize=13)
    sns.regplot(x=column_name, y=target, data=data, scatter=False, color='b')
    plt.show()


def plot_scatter(data, column_name, target):
    fig, ax = plt.subplots()
    ax.scatter(x=data[column_name], y=data[target])
    plt.xlabel(column_name, fontsize=13)
    plt.ylabel(target, fontsize=13)
    plt.show()


def plot_box(data, column_name, target):
    data = pd.concat([data[target], data[column_name]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=column_name, y=target, data=data)
    fig.axis(ymin=0, ymax=800000)


def plot_category_columns(data, limit_bars=10):
    category_cols = data.select_dtypes(include=['object'])

    for col in category_cols:
        plt.figure(figsize=(12, 6))
        if len(data[col].unique()) < limit_bars:
            sns.countplot(x=col, data=data)
        else:
            sns.countplot(y=col, data=data)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Arguments:
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.
        title : string
            Title for the chart.
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects
    """

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid('on')
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
