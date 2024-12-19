SeaCoral framework documentation!
=================================

This framweork can be used to plot datasets in pythong without repeating
the same code over and over again.
The framework is based on Seaborn_ and Matplotlib_ library and is designed to be
flexible and easy to use.
The plots can be easily tuned to more spohisitcated with custom setups if
necessary.
The main usage of this library is for scientific pubblications where given
the same data sometimes you just wanted to move the legend or change the
font without having to look into the codebase or hack the plot.

.. _Seaborn: https://seaborn.pydata.org/
.. _Matplotlib: https://matplotlib.org/

Using this framework
--------------------

:doc:`Installing`
    How to install this tool in your machine.

:doc:`CLIusage`
    How to use this tool from the CLI without need to write code.

:doc:`APIusage`
    How to include this library into your code for easy plotting.

:doc:`Customization`
    How to include new functions into the framework to personalize your plots.

:doc:`Features`
    List of supported features and how to use them.

:doc:`Examples`
    A broader list of examples of how to use the framework.

Dataset Manipulation
--------------------

:doc:`Dataset`
    How to datasets are loaded into the framwork

:doc:`DatasetOperations`
    How to manipulate the datasets

:doc:`DatasetForking`
    How to copy and modify an already loaded dataset

Plot Generation
---------------

:doc:`Plot`
    How to generate simple plots

:doc:`PlotLevelCustomization`
    Available customization options that applies to all generated plots

:doc:`FigureLevelCustomization`
    Available customization options applicable to plots individually

:doc:`UserDefinedFunctions`
    How to include new functions into the framework to personalize your plots.

Development
-----------

:doc:`Contributing`
    How to contribute changes to the framework.

:doc:`DevelopmentGuidelines`
    Guidelines the theme developers use for developing and testing changes.

:doc:`Changelog`
    The framework development changelog.

.. Hidden TOCs

.. toctree::
   :caption: Usage
   :maxdepth: 2
   :hidden:

   Installing
   CLIusage
   APIusage
   Customization
   Features
   Examples

.. toctree::
   :caption: Dataset
   :maxdepth: 2
   :hidden:

   Dataset
   DatasetOperations
   DatasetForking

.. toctree::
   :caption: Plot
   :maxdepth: 2
   :hidden:

   Plot
   PlotLevelCustomization
   FigureLevelCustomization
   UserDefinedFunctions

.. toctree::
   :caption: Development
   :maxdepth: 2
   :hidden:

   Contributing
   DevelopmentGuidelines
   Changelog
   modules
