.. _src-code_data-preparation:

=====================================
Data Preparation
=====================================

When preparing the input data for your model application, you use the following two modules: ``template_creation.py``
and ``data_loading.py``. For the code of these modules, see their respective pages below.

Before the data gets processed by the model, you also have to :ref:`set the Model Configuration<workflow_model-configuration>`.
Regarding this, the options of scaling, clustering and time averaging are elaborated upon in their respective documentation.

Template Creation
=====================================

The module ``template_creation.py`` is used to create templates for the model configuration and input data directory.
Explanation on the methods for these can be found :ref:`here for the model templates <workflow_create-model-templates>` and
:ref:`here for the input data templates <workflow_create-data-templates>`.

.. automodule:: adopt_net0.data_preprocessing.template_creation
    :members:
    :exclude-members:

Data Loading
=====================================

The module ``data_loading.py`` is used to load data into your input data folder from different sources (e.g., from an API,
from the repository of this model, or from external datasets). Explanation on which method is useful for which datatype
can be found :ref:`here <workflow_load-data>`.

.. automodule:: adopt_net0.data_preprocessing.data_loading
    :members:
    :exclude-members:

Model Configuration
=====================================

When preparing your data for the model, you can also specify options to reduce the complexity of the model in the
Model Configuration (set in ``ConfigModel.json``, see :ref:`here<workflow_model-configuration>`). These options are
elaborated :ref:`here<model_configuration>`
