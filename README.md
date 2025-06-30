# RobCoClust

This directory contains the code to reproduce the results presented in the article *Cell-TRICC: a Model-Based Approach 
to Cellwise-Trimmed Co-Clustering*, submitted to the Journal of Computational and Graphical Statistics.

```TCoClust``` is the Python package developed for this paper, implementing the Cell-TRICC method and additional tools 
(including plotting and functions for simulation).

Three Jupyter notebooks are included, one (```example_notebook.ipynb```) to showcase the main functionalities and basic 
usage of the Python package, and others to reproduce the results presented in the manuscript. 
The results of the simulations are also included, under ```data/simulations```. 
Simulations rely on the code in ```simulations``` and the '```_caller.py```' scripts.

## Setup

A ```requirements.txt``` file is provided, containing the minimal dependencies needed to run the code and reproduce the 
results. An environment can be built by simply running ```$ conda create --name <env> --file``` in a terminal, where 
```<env>``` is to be replaced with a desired name for the conda environment. Conda should be installed on the machine.

## Project content and structure
The main content of the root directory is:
* TCoClust package
* Jupyter notebooks to show package usage and reproduce the figures included in the paper and appendix, namely:
  * `example_notebook.ipynb`: example of usage of the package
  * `simulations_extensive.ipynb`: notebook to reproduce the simulation results
  * `simulations_model_selection.ipynb`: notebook to reproduce the model selection simulations
  * `sanctions_sanitized.ipynb`: notebook to reproduce the results of the real data application
* Simulation scripts, organized as follows:
    * `simulations`: library of main simulation functions
    * `simulations_new_caller.py`, `simulations_roco_tricc_caller.py`, `simulations_model_selection_caller.py`: scripts to execute the simulations in parallel
* `data` directory containing intermediate results from the simulations and the application (**to load the results without re-running everything, which would require several hours, depending on parallelization and computing power**)

Below a more detailed tree representation of the directory structure:
```
.
|-- README.md
|-- TCoClust
|   |-- __init__.py
|   |-- cell_tricc.py
|   |-- class_defs.py
|   |-- config.py
|   |-- metrics.py
|   |-- model_selection.py
|   |-- normal_utils.py
|   |-- plots.py
|   |-- poisson_utils.py
|   |-- roco_tricc.py
|   `-- utils.py
|-- data
|   |-- do_not_share
|   |   `-- contTab4d_wNames_MASKED.csv
|   |-- sanctions_results
|   |   `-- model_selection_all_alpha.csv
|   `-- simulations
|       |-- extensive
|       |-- model_selection_larger_beta
|       `-- poisson_simul_params
|-- example_notebook.ipynb
|-- requirements.txt
|-- sanctions_sanitized.ipynb
|-- simulations
|   |-- __init__.py
|   |-- simulations_model_selection.py
|   |-- simulations_new.py
|   `-- simulations_roco_tricc.py
|-- simulations_extensive.ipynb
|-- simulations_model_selection.ipynb
|-- simulations_model_selection_caller.py
|-- simulations_new_caller.py
`-- simulations_roco_tricc_caller.py

```