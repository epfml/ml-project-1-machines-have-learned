# Machine Learning Project README - Part 1

## Authors:
- Ilias Marwane Merigh
- Youssef Belghmi
- Hamza Morchid

### Project Overview

This README provides an overview of the Machine Learning Project 1, its structure, and how to test the implemented code. The project is divided into multiple files, each serving a specific purpose.

### File List

1. **README.md**
   - This file provides an overview of the project and instructions on how to test the implementations.

2. **implementations.py**
   - This script contains 6 functions fully commented as specified in the project statement. This module is entirely independent of the rest of the project.

3. **run.py**
   - This script generates the submitted .csv file for AIcrowd competition. To test it, simply execute the command: `python run.py`. The script will take some time to complete, and you will see a progress report in the console. Note that the script loads data from the local root path "data/dataset_to_release_2."

4. **best_model.ipynb**
   - This Jupyter notebook is the core of our project. It includes code, details about each algorithm tried, that led to our final algorithm used in `run.py`. Some cells may take a while to run. However, we have ensured that the notebook is fully executed, and all cells are evaluated.

5. **comparison.ipynb**
   - In this notebook, we compare our model with the one from `sklearn`. We make claims and draw conclusions. We have separated this comparison into a distinct file to emphasize that `best_model.ipynb` has no dependencies with other libraries. Just like the previous notebook, all cells are fully executed and evaluated.

6. **report.pdf**
   - This document contains detailed findings. It explains our assumptions, claims, conclusions, findings, and any mistakes made during the project. Unlike `best_model.ipynb`, this document does not contain any code but provides an in-depth explanation of our thought process and intuition behind implementing such a model.

### Testing Instructions

To test our project, follow these steps:

1. Execute `python run.py` to generate the AIcrowd submission .csv file. Be patient, as it might take some time.

2. Refer to the `best_model.ipynb` and `comparison.ipynb` notebooks for detailed insights and comparisons.

3. Review the `report.pdf` for a comprehensive explanation of our findings.

If you have any questions or encounter issues, please don't hesitate to reach out to the authors for assistance. We hope you find our project informative and insightful.
