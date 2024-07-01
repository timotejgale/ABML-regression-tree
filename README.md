# ABML regression tree

ABTree (ABML regression tree) tool extends the CART algorithm and implements the argument-based regression tree (ABRT) method as a set of Python scripts. Furthermore, it provides the corresponding knowledge refinement loop implementation. ABTree is realized as an object-oriented program, in addition to standalone usage it may also be used as a library. The tool supplies various methods for parsing data and associated ABRT arguments. Evaluation methods are available.

## Setup

The installation process assumes a Python3 installation is available. To run the script in the standalone mode, run the following commands:

```
# Create a virtual environment.
python -m venv venv

# Activate the virtual environment.
. ./venv/bin/activate

# Install the dependencies.
pip install -e .

# Run the demo script.
python abtree.py
```

## Additional information and citing

For a comprehensive understanding of the methodologies and principles behind this tool, please refer to our paper:

Gale, T., & Guid, M. (2024, May). Argument-Based Regression Trees. In _2024 47th MIPRO ICT and Electronics Convention (MIPRO)_ (pp. 1-6). IEEE.

This paper includes in-depth discussions on the methodologies, theoretical frameworks, and performance benchmarks of the tool. If you utilize or build upon this tool in your work, we would appreciate if you cite our paper. BibTeX format:

```
@inproceedings{gale2024argument,
  title={Argument-Based Regression Trees},
  author={Gale, Timotej and Guid, Matej},
  booktitle={2024 47th MIPRO ICT and Electronics Convention (MIPRO)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
