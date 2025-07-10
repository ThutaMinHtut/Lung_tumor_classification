# AI-Based Classification of Lung Tumors in CT Scans with Synthetic Data Augmentation

## Project Description
This project explores synthetic data augmentation techniques to improve lung tumor classification from CT scans using deep learning.  
It focuses on generating full 2D CT slices (preserving ribcages, anatomical structures, and imaging artifacts) using Conditional GANs (cGANs) to enrich limited and imbalanced datasets.  
The project compares Baseline, Traditional Augmentation (MONAI), and GAN-based augmentation methods on multiclass tumor classification performance given a small dataset.

## Dataset
- Kaggle CT Scan Images for Lung Cancer Dataset: [Link Here](https://www.kaggle.com/datasets/dishantrathi20/ct-scan-images-for-lung-cancer/data)
- External Supplement Dataset:[PETCTDX](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
```
Dataset folder structure:
[PLACEHOLDER]/
└── LungcancerDataSet/
    ├── test/
    │   ├── adenocarinoma/
    │   ├── BenginCases/
    │   ├── MalignantCases/
    │   ├── normal/
    │   ├── large.cell.carcinoma/
    │   └── squamous.cell.carcinoma/
    ├── train/
    │   ├── adenocarinoma/
    │   ├── BenginCases/
    │   ├── MalignantCases/
    │   ├── normal/
    │   ├── large.cell.carcinoma/
    │   └── squamous.cell.carcinoma/
    └── valid/
        ├── adenocarinoma_left.lower.lobe_T2_N0_M0_lb/
        ├── BenginCases/
        ├── MalignantCases/
        ├── normal/
        ├── large.cell.carcinoma_left.hilum_T2_N2_M0_llla/
        └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_llla/
  ```



Replace [PLACEHOLDER] with the actual file names after cloning repo. </br>
Utilizes PyTorch Version 2.5.1 & CUDA 12.6.1
