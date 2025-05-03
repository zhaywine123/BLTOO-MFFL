# BLTOO-MFFL
BLTOO-MFFL: Automated and Adversarially Robust Deep Learning for SAR Image Classification by Bi-Level Three-Objective Optimization and Multi-Feature Fusion Loss

Training and testing configurations for FUSAR and OpenSarShip are provided. If you need to test, please download the corresponding dataset and put it in the './data/' folder.

You can directly use BLTOO-MFFL_all_in_one.ipynb for testing. Please note that if you use the original configuration (without modifying the KL calculation), please do not set a batch_size that is too large. We recommend a maximum of 32, which may result in excessive computation time consumption.


Sorry for not providing the environment package directly. Users can build the pytorch environment and install the pymoo package to implement the evolution process. Generally, it can meet the needs of use. If it prompts that the package is missing, please run pip install <'package name'>.
