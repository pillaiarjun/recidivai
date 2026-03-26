#!/usr/bin/env python3
"""Initialize all project notebooks with valid nbformat skeletons."""
import json
from pathlib import Path

EMPTY_NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.6"
        }
    },
    "cells": []
}

notebooks = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_bronze_ingestion.ipynb",
    "notebooks/03_silver_pipeline.ipynb",
    "notebooks/04_gold_features.ipynb",
    "notebooks/05_modeling.ipynb",
]

for path_str in notebooks:
    p = Path(path_str)
    p.write_text(json.dumps(EMPTY_NB, indent=1))
    print(f"✓ Initialized {p}")

print("All notebooks initialized.")
