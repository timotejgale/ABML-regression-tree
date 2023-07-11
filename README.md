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