# SAGS-Net
We have uploaded the dataset processing code, if you find it useful, please cite the following article.
@article{LIU2023105145,
title = {A Soft-Attention Guidance Stacked neural Network for neoadjuvant chemotherapy’s pathological response diagnosis using breast dynamic contrast-enhanced MRI},
journal = {Biomedical Signal Processing and Control},
volume = {86},
pages = {105145},
year = {2023},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2023.105145},
url = {https://www.sciencedirect.com/science/article/pii/S1746809423005785},
author = {Tianyu Liu and Hong Wang and Shengpeng Yu and Feiyan Feng and Jun Zhao},
keywords = {Breast cancer, Dynamic contrast-enhanced MRI, Soft attention, Neoadjuvant chemotherapy, Pathological complete response},
abstract = {Pathological complete response (pCR) to Neoadjuvant chemotherapy (NACT)is a significant clinical indicator for diagnosing patient outcomes and overall survival. However, the current diagnosis of pCR suffers from two limitations: firstly, it requires an invasive biopsy, intensifying the patient’s pain and risk of infection. Secondly, current radiomics-based computer methods inefficiently utilize intrinsic characteristics of medical images. In this paper, we present a Soft-Attention Guidance Stacked neural Network (SAGS-Net) to address these limitations and predict the NACT responses from breast dynamic contrast-enhanced MRI (DCE-MRI). Particularly, We first design a self-adapting feature selection strategy to extract peritumoral features while excluding outside noises accurately. Then, SAGS-Net is built by stacking position-based spatial models that generate discriminative feature representation. Each stacked model inside has its semantic feature branch as a control gate for pCR-related feature selection. The semantic feature branch combined with the residual learning mechanism as a feature selector enhances valuable information and suppresses redundant information, simulating the process of radiologists making a clinical diagnosis based on domain knowledge. Finally, the incremental stacked network architecture assisted with the soft-attention strategy can gradually refine attention-aware features in complex DCE-MRI sequences. Experimental results based on real-world clinical datasets confirmed that the proposed SAGS-Net obtain superior performance with 93% AUC, and it provides a new way that leverage DCE-MRI sequences to predict NACT responses.}
}
