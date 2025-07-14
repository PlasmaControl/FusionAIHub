# FusionAIHub

A general fusion hub for the Princeton cluster designed to standardize fusion machine learning processes for the plasma control group at Princeton University.

## Purpose

This repository serves as a centralized platform for fusion-related machine learning workflows, providing standardized tools, processes, and methodologies for plasma control research at Princeton.

## Setup

Go to your scratch directory while you are on the HEAD node (so you need internet access, which computing nodes do not have).

We will be using Python 3.12.

/scratch/gpfs/[username]

In your scratch directory, run
```bash
git clone git@github.com:PlasmaControl/FusionAIHub.git
cd FusionAIHub
git switch foundation25
module load anaconda3/2024.10
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install uv
uv sync
```

## Contact

For more information, please contact
- **Azarakash Jalalvand**
- **Peter Steiner** 
- **Kouroche Bouichat**
- **Nathaniel Chen**
