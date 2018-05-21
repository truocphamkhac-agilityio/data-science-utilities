======================
Data Science Utilities
======================


.. image:: https://img.shields.io/pypi/v/data_science_utilities.svg
        :target: https://pypi.python.org/pypi/data_science_utilities

.. image:: https://img.shields.io/travis/truocphamkhac/data-science-utilities.svg
        :target: https://travis-ci.org/truocphamkhac/data-science-utilities

.. image:: https://readthedocs.org/projects/data-science-utilities/badge/?version=latest
        :target: http://data-science-utilities-python.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Data Science utilities in python.


* Free software: MIT license
* Documentation: http://data-science-utilities-python.readthedocs.io.


Features
--------

* Missing Data Statistic::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> # make statistic
    >>> missing_data = data_science_utilities.missing_data_stats(df)
    >>> # display statistic
    >>> missing_data


* Read CSV files from path::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> train_path = '../data/raw/train.csv'
    >>> test_path = '../data/raw/test.csv'
    >>>
    >>> X_train, X_test = data_science_utilities.read_csv_files(train_path, test_path)


* Plotting distribution normal::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_dist_norm(dist, 'distribution normal')


* Plotting correlation matrix::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_corelation_matrix(data)


* Plotting top attributes correlation matrix::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_top_corelation_matrix(data, target, k=10, cmap='YlGnBu')


* Plotting attributes by scatter chart::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_scatter(data, column_name, target)


* Plotting attributes by box bar::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_box(data, column_name, target)


* Plotting category by box bar::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_category_columns(data, limit_bars=10)


* Generate a simple plot of the test and traning learning curve::

    >>> from data_science_utilities import data_science_utilities
    >>>
    >>> data_science_utilities.plot_learning_curve(estimator, title, X, y, ylim=None,
    >>>                     cv=None, train_sizes=np.linspace(.1, 1.0, 5))


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
