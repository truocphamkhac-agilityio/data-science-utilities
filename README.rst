============
ML Utilities
============


ML utilities in python.


* Free software: MIT license
* Documentation: https://gitlab.asoft-python.com/bgh/data-science/ml-utilities.


Features
========

Missing Data Statistic
----------------------

.. code:: python

    from ml_utilities import ml_utilities

    # make statistic
    missing_data = ml_utilities.missing_data_stats(df)

    # display statistic
    missing_data


Plotting distribution normal
----------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_dist_norm(dist, 'distribution normal')


Plotting correlation matrix
---------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_corelation_matrix(data)


Plotting top attributes correlation matrix
------------------------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_top_corelation_matrix(data, target, k=10, cmap='YlGnBu')


Plotting attributes by scatter chart
------------------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_scatter(data, column_name, target)


Plotting attributes by box bar
------------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_box(data, column_name, target)


Plotting category by box bar
----------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_category_columns(data, limit_bars=10)


Generate a simple plot of the test and traning learning curve
-------------------------------------------------------------

.. code:: python

    from ml_utilities import ml_utilities

    ml_utilities.plot_learning_curve(estimator, title, X, y, ylim=None,
                        cv=None, train_sizes=np.linspace(.1, 1.0, 5))


Development
===========

Powered by Team IO.
