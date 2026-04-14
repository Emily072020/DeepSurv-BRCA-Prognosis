# DeepSurv-BRCA-Prognosis
🧬 DeepSurv-BRCA: Deep Survival Analysis for Breast Cancer
This repository features a DeepSurv implementation—a deep learning-based Cox proportional hazards model—designed to predict the prognosis (PFI) of Breast Invasive Carcinoma (BRCA) patients using TCGA transcriptomic data.

🎯 Project Highlights
-> Robust Performance: Achieved a test set C-index of 0.7446, significantly outperforming single-gene biomarkers.

-> Precision Engineering: Developed a two-stage feature selection pipeline (Variance filtering + Cox/Correlation screening) to distill 20,000+ genes into a 500-gene prognostic signature.

-> Reproducibility: Locked random seeds and used deterministic cuDNN algorithms to ensure 100% consistent results across different platforms.

-> Deployment Ready: Built a complete inference pipeline, including automated data normalization (`StandardScaler`) and risk stratification.

📊 Key Results: Clinical Stratification
The model demonstrates strong clinical utility by effectively stratifying patients into distinct risk groups.

-> P-value < 0.0001: Highly significant divergence between High-Risk (Top 25%) and Low-Risk groups.

-> Stability: The stratification remains robust across both median-split and top-quartile thresholds.

🛠️ Usage & Structure
Quick Start

Python

`# 3-line inference for new patient data`

`from scripts.predict import predict_patient_risk`

`results = predict_patient_risk(new_gene_expression_data)`

`# Returns: [Risk_Score, Risk_Level]`

File Navigation

-> `train_DeepSurv.py`: Modular training script (Data cleaning -> Feature Selection -> Training).

-> `DeepSurv_BRCA_Best_0.7446.pth`: Pre-trained neural network weights.

-> `reproduce_results.ipynb`: Notebook for generating KM curves and model evaluation.

-> `artifacts/`: Pickled `scaler` and `gene_list` for consistent data preprocessing.

Author: Jinhua Liu (Bioinformatics PhD)

Core Stack: PyTorch, Lifelines, Scikit-learn, Pandas

License: MIT
