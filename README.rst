.. list-table::
:stub-columns: 1
  *   - docs
      - | .. image:: [https://readthedocs.org/projects/fusionaihub/badge/?version=latest](https://www.google.com/search?q=https://readthedocs.org/projects/fusionaihub/badge/%3Fversion%3Dlatest)
        :target: [https://fusionaihub.readthedocs.io/en/latest/?badge=latest](https://www.google.com/search?q=https://fusionaihub.readthedocs.io/en/latest/%3Fbadge%3Dlatest)
        :alt: Documentation Status
  *   - tests
      - | .. image:: [https://github.com/PlasmaControl/FusionAIHub/actions/workflows/tests.yml/badge.svg](https://www.google.com/search?q=https://github.com/PlasmaControl/FusionAIHub/actions/workflows/tests.yml/badge.svg)
        :target: [https://github.com/PlasmaControl/FusionAIHub/actions/workflows/tests.yml](https://www.google.com/search?q=https://github.com/PlasmaControl/FusionAIHub/actions/workflows/tests.yml)
        :alt: Unit Tests
  *   - package
      - | .. image:: [https://img.shields.io/pypi/v/fusionaihub.svg](https://www.google.com/search?q=https://img.shields.io/pypi/v/fusionaihub.svg)
        :target: [https://pypi.python.org/pypi/fusionaihub](https://www.google.com/search?q=https://pypi.python.org/pypi/fusionaihub)
        :alt: PyPI Package
        | .. image:: [https://img.shields.io/pypi/pyversions/fusionaihub.svg](https://www.google.com/search?q=https://img.shields.io/pypi/pyversions/fusionaihub.svg)
        :target: [https://pypi.python.org/pypi/fusionaihub](https://www.google.com/search?q=https://pypi.python.org/pypi/fusionaihub)
        :alt: Supported Python Versions
  *   - quality
      - | .. image:: [https://codecov.io/gh/PlasmaControl/FusionAIHub/branch/main/graph/badge.svg](https://www.google.com/search?q=https://codecov.io/gh/PlasmaControl/FusionAIHub/branch/main/graph/badge.svg)
        :target: [https://codecov.io/gh/PlasmaControl/FusionAIHub](https://www.google.com/search?q=https://codecov.io/gh/PlasmaControl/FusionAIHub)
        :alt: Codecov
  *   - meta
      - | .. image:: [https://img.shields.io/github/license/PlasmaControl/FusionAIHub.svg](https://www.google.com/search?q=https://img.shields.io/github/license/PlasmaControl/FusionAIHub.svg)
        :target: [https://github.com/PlasmaControl/FusionAIHub/blob/main/LICENSE](https://www.google.com/search?q=https://github.com/PlasmaControl/FusionAIHub/blob/main/LICENSE)
        :alt: License
        | .. image:: [https://img.shields.io/github/issues/PlasmaControl/FusionAIHub.svg](https://www.google.com/search?q=https://img.shields.io/github/issues/PlasmaControl/FusionAIHub.svg)
        :target: [https://github.com/PlasmaControl/FusionAIHub/issues](https://www.google.com/search?q=https://github.com/PlasmaControl/FusionAIHub/issues)
        :alt: GitHub Issues

# FusionAIHub

A general fusion hub for the Princeton cluster designed to standardize fusion machine learning processes for the plasma control group at Princeton University.

## Purpose

This repository serves as a centralized platform for fusion-related machine learning workflows, providing standardized tools, processes, and methodologies for plasma control research at Princeton.

## Setup

Go to your scratch directory while you are on the HEAD node (so you need internet access, which computing nodes do not have).

We will be using Python 3.12 and `uv` as a package manager. Since `uv` isn't on Stellar, for now we will install it via pip.

`/scratch/gpfs/[username]`

In your scratch directory, run:

.. code-block:: bash

```
git clone git@github.com:PlasmaControl/FusionAIHub.git
cd FusionAIHub
git switch foundation25
module load anaconda3/2024.10
python -m venv .venv
conda deactivate
source .venv/bin/activate
pip install --upgrade pip
pip install uv
uv sync
```

From now on, whenever you go into the repo, all you need to do is to run:

.. code-block:: bash

```
source .venv/bin/activate
```

## Contact

For more information, please contact:

  * **Azarakash Jalalvand**
  * **Peter Steiner**
  * **Kouroche Bouchiat**
  * **Nathaniel Chen**