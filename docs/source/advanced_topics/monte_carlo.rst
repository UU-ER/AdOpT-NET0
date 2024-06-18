..   _monte_carlo:

Monte Carlo analysis
=====================================
Monte Carlo analysis is a statistical technique that uses random sampling and statistical methods to estimate uncertainty
and variability in system performance and economic outcomes. In this framework, Monte Carlo analysis is applied specifically
to economic input parameters, including technology and network capital expenditures (CAPEX) as well as import and export
prices. To perform the Monte Carlo analysis, you first need to specify the number of simulations in the ``ConfigModel.json``
file. Next, you can select the sampling method in the ``ConfigModel.json`` file. The framework includes two different
sampling methods:

1. **Normal Distribution Sampling**: Parameters are varied based on a normal distribution, using a standard deviation
   provided by the user in the ``ConfigModel.json`` file. For this method, you can also specify the components you want to
   perform the analysis on.

2. **Uniform Distribution Sampling**: Parameters are varied based on a uniform distribution between minimum and maximum
   values provided by the user in a ``MonteCarlo.csv`` file. In order to create this file, you pass the path to the
   input data folder through the :func:`create_montecarlo_template_csv` method in the main module (``main.py``) (similar
   to creating :ref:`the model templates<workflow_create-model-templates>` or the :ref:`data templates<workflow_create-data-templates>`.
   For documentation of this :func:`create_montecarlo_template_csv` function, see the :ref:`source code documentation<src-code_data-preparation>`.



