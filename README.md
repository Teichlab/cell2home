# Cell2home

Cell2home is a tool to computationally infer cell-cell homing interactions and make predictions of which cells attract each other in single-cell datasets. For example, cell2home predicts which cell types (such as vascular, fibroblast and other immune cells) attract immune cell subsets based on a curated chemoattractant receptor-ligand network. Cell2home first computes chemoattractant receptor-ligand interaction scores and then summarises these homing signals across receptors into a homing affinity score between a migrating immune cell and an attracting cell, repeating this across all migrating and attracting cell types to achieve a global immune cell homing landscape.

<p align="center"><img src="https://github.com/Teichlab/cell2home/blob/main/figure.png" alt="Label evolution" width="80%" ></p>

## Installation

For now, the package needs to be installed from GitHub:

```bash
git clone https://github.com/Teichlab/cell2home
cd cell2home
pip install .
```

## Usage and Documentation

Please refer to [ReadTheDocs](https://cell2home.readthedocs.io/en/latest/), which comes with a demo notebook and function docstrings.

## Citation

Coming soon!
