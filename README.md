# TCoClust

```TCoClust``` is a Python package implementing robust Trimmed Co-Clustering methods, including a method for cellwise 
trimmed co-clustering (Cell-TRICC, to be published), a method for row and columnwise trimmed co-clustering (RoCo-TRICC, 
published in Fibbi, E., Perrotta, D., Torti, F., Van Aelst, S., Verdonck, T. "Co-clustering contaminated data: a robust 
model-based approach." *Adv Data Anal Classif* 18, 121–161 (2024). https://doi.org/10.1007/s11634-023-00549-3),
as well as additional tools (including model selection, plotting, and other utility functions).

An example Jupyter notebook (```example_notebook.ipynb```) is included to showcase the main functionalities and basic 
usage of the Python package.

## Setup

A ```requirements.txt``` file is provided, containing the minimal dependencies.

## Project content and structure
The main content of the root directory is:
* **TCoClust package**, including the following main modules:
  * ```cell_tricc.py``` implementing cellwise trimmed co-clustering
  * ```roco_tricc.py``` implementing row and columnwise trimmed co-clustering
  * ```model_selection.py``` to perform model selection via ICL criteria and trimming monitoring
  * ```plots.py``` containing some plotting functions and exploratory tools
  * ```metrics.py``` for some metrics commonly used in co-clustering
* A **Jupyter notebook** to show package usage

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
|-- example_notebook.ipynb
`-- requirements.txt

```
