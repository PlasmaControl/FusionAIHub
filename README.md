# FusionAIHub

A general fusion hub for the Princeton cluster designed to standardize fusion machine learning processes for the plasma control group at Princeton University.

## Purpose

This repository serves as a centralized platform for fusion-related machine learning workflows, providing standardized tools, processes, and methodologies for plasma control research at Princeton.

## Team

This project is led by:
- **Egemen Kolemen**
- **Azarakash Jalalvand**
- **Peter Steiner** 
- **Kouroche Bouichat**
- **Nathaniel Chen**

## Setup

When you are in the root directory, you can run the following command to activate the virtual environment:
```bash
module load anaconda3/2024.10
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install uv
uv sync
```