# Welcome to the Fusion Artificial Intelligence Hub (FAITH) - Princeton Plasma Control Group

A general fusion hub for the Princeton cluster designed to standardize fusion machine learning processes at Princeton University.

## Purpose

FAITH serves as a centralized platform for fusion-related machine learning workflows, providing standardized tools, processes, and background optimization for plasma control research at Princeton. This way, you can worry less about dataset/gpu optimiation and standardization. This platform primarily serves as a fast machine-learning trainer, but can also be used for general data analysis for fusion data stored at Princeton.

## General Usage

1. Clone this repository onto anywhere on Stellar
2. Get the location of this repository (you can find it by typing `pwd` in the terminal)
3. In your main project's virtual environment or anaconda environment, type `pip install </path/to/FusionAIHub>`
4. You can find tutorials on how to use this package in `docs`, `example` and `notebooks`

`TODO: We will include this as a pip package in the future`

## Setup [for developers]

Go to your scratch directory while you are on the HEAD node (so you need internet access, which computing nodes do not have).

We will be using Python 3.12 and uv as a package manager. Since uv isn't on Stellar, for now we will install it via pip. First head over to your scratch directory with
```bash
cd /scratch/gpfs/[username]
```

In your scratch directory, run
```bash
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

From now on, whenever you go into the repo, all you need to do is to run
```bash
source .venv/bin/activate
```

## Contact

For more information, please contact
- **Peter Steiner** ([ps9551@princeton.edu](mailto:ps9551@princeton.edu))
- **Nathaniel Chen** ([nathaniel@princeton.edu](mailto:nathaniel@princeton.edu))
- **Kouroche Bouchiat** ([bouchiat@princeton.edu](mailto:bouchiat@princeton.edu))
- **Azarakash Jalalvand** ([aj17@princeton.edu](mailto:aj17@princeton.edu))
- **Egemen Kolemen** ([ekolemen@princeton.edu](mailto:ekolemen@princeton.edu))
