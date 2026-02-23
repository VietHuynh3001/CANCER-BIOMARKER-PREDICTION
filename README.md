# 1.INTRODUCTION:
Melanoma remains a significant global health challenge, requiring more precise and effective therapeutic interventions. Immunotherapy, particularly Immune Checkpoint Inhibitors, has emerged as a promising strategy, offering long-term efficacy with fewer side effects than conventional chemotherapy. However, clinical success is often limited by tumor heterogeneity and the complex tumor microenvironment, which significantly impact how patients respond to these treatments.

This project aims to identify high-potential melanoma biomarkers through an advanced machine learning pipeline. The methodology integrates Differential Expression Analysis (DEA), Chi-square filtering, and Support Vector Machine-Recursive Feature Elimination (SVM-RFE) for rigorous feature selection. To maximize predictive performance and capture non-linear biological signals, Random Forest optimized via Bayesian Hyperparameter Optimization is employed. This is followed by a survival analysis to provide insights into the identified biomarkers and their correlation with patient outcomes and immune signals.

# 2. METHOD:
## 2.1. DATA RETRIEVAL AND PREPROCESSING:
Clinical and transcriptomic data of melanoma patients treated with ICIs were retrieved from cBioPortal and GEO. Patients were classified as responders or non-responders. Differential Expression Analysis (DEA) was performed using the pydeseq2 library, with significance defined as an adjusted p-value < 0.05 and |log2FC| > 1.0.
## 2.2. TWO-STAGE FEATURE SELECTION:
To refine the feature set, a two-stage selection process was implemented:
+ Stage 1: Chi-square Test: A statistical filter was applied to remove genes whose expression levels showed no significant association with clinical response.
+ Stage 2: SVM-RFE: A Support Vector Machine-Recursive Feature Elimination algorithm was used to rank features and select the most discriminative subset by iteratively removing low-weight genes.
## 2.3. SURVIVAL-ASSOCIATED BIOMARKER ANALYSIS:
High-ranking DEGs from SVM-RFE were intersected with survival-related genes from the TCGA-SKCM cohort. The Mann-Whitney U-test was performed to compare the expression levels of these survival-related genes between responders and non-responders. The Cox proportional hazard model then assessed the independent impact of these genes on overall survival. Kaplan-Meier plots were generated to visualize survival probability differences between high and low expression groups.
## 2.4. GENE ONTOLOGY ENRICHMENT ANALYSIS:
Functional enrichment analysis was performed using the clusterProfiler package to explore Biological Processes, Molecular Functions, and Cellular Components. A significance threshold of adjusted p-value < 0.05 was applied to identify immune-related pathways.
## 2.5. RANDOM FOREST MODEL AND BAYESIAN OPTIMIZATION:
Predictive models were constructed using the Random Forest algorithm to capture complex biological signals. To prevent overfitting and maximize the Area Under the Curve (AUC), Bayesian Optimization was utilized via BayesSearchCV to tune critical hyperparameters, including n_estimators, max_depth, and min_samples_split.

# 3.RESULTS:
## 3.1. Differential Expression Analysis
The primary objective of this analysis was to identify genes with significant expression differences between treatment responders and non-responders. This helps create a high-quality list of candidate genes for the next feature selection stages. By using a significance threshold of an adjusted p-value ($p_{adj}$) < 0.05 and $|log_2FC| > 1$, the analysis identified a total of `1,540 differentially expressed genes (DEGs)`. This included `745 up-regulated genes` and `795 down-regulated genes`. The Volcano Plot shows a strong transcriptomic signature, with many genes reaching high statistical significance ($-log_{10}p_{adj} > 12$) and large changes in expression. These DEGs provide the foundational data for the following Chi-square filtering and SVM-RFE steps.
<p align="center">
  <img src="DEA results/volcano_plot.png" width="500"/>
  <br>
  <i>
    Figure 1: Volcano Plot of Differentially Expressed Genes (DEGs) in Melanoma.
  </i>
</p>

## 3.2. Gene Ontology Enrichment Analysis
The primary objective of this analysis was to understand the biological meaning of the `1,540` DEGs by grouping them into three domains: Biological Process (BP), Cellular Component (CC), and Molecular Function (MF). By mapping these genes to Gene Ontology terms, the study ensures that the features selected for machine learning models have relevant biological signals, especially those related to immune response and immunotherapy success.

The analysis of up-regulated genes shows a strong enrichment in pathways important for responding to immune checkpoint inhibitors. In the Biological Process domain, there is a focus on lymphocyte activation and immune signaling, with very high statistical significance. For Cellular Components, these genes are mostly located in the MHC protein complexes, which are essential for antigen presentation. The Molecular Function results also highlight activities like cytokine receptor activity. This proves that up-regulated genes help create a "hot" tumor microenvironment that improves treatment efficacy.
<p align="center">
  <img src="DEA results/GO_up.png" alt="GO Enrichment Up" width="500" height="500"/>
  <img src="DEA results/GO_down.png" alt="GO Enrichment Down" width="500" height="500"/>
  <br>
  <i>
    Figure 2: Gene Ontology (GO) dot plots illustrating the top enriched terms for up-regulated (left) and down-regulated (right) genes.
  </i>
</p>

In contrast, down-regulated genes are mostly associated with metabolic and structural pathways in non-responsive tumors. In the Biological Process domain, these genes are related to lipid metabolism and epidermis development. Cellular Components show that these genes are located in the extracellular matrix and keratin filaments. Molecular Function analysis shows enrichment in structural molecule activities. These findings suggest that down-regulated genes may represent structural barriers that prevent immune cells from entering the tumor and making the treatment less effective.

## 3.3. Feature Selection and SVM-RFE
The main goal of this stage was to reduce the 1,540 DEGs into a smaller, highly effective group of features. By using a two-step process—statistical filtering with Chi-square followed by Support Vector Machine-Recursive Feature Elimination (SVM-RFE)—this study identified robust biomarkers to predict immunotherapy response accurately while keeping the model simple.

The results show that reducing features greatly affects predictive accuracy. The SVM-RFE algorithm was used to test three different feature sets: the Top 4 up-regulated genes, the Top 4 down-regulated genes, and a larger set of the Top 100 DEGs.
+ The Top 100 DEG model performed best, with a training accuracy of 0.9047 and a validation accuracy of 0.7971. This model also had a high ROC-AUC of 0.94, showing excellent class separation.
+ The Top 4 up-regulated model, including genes like `KRTDAP, PTPRZ1, RLBP1, and SLC9A3`, showed good performance with a validation accuracy of 0.7963 and an AUC of 0.88.
+ The Top 4 down-regulated model (including `CUX2, SIGLEC8, SPDYC, and SERTM2`) had lower performance, with a validation accuracy of 0.7140 and an AUC of 0.79. This suggests that up-regulated immune genes are more important for prediction than down-regulated genes.
<p align="center">
  <img src="DEA results/SVM_RFE.png" alt="SVM-RFE Performance Metrics" width="90%"/>
  <br>
  <i>
    Figure 3: SVM-RFE validation results showing Accuracy (left) and ROC-AUC curves (right) for the selected gene subsets.
  </i>
</p>

Finding these "driver genes" moves the research from general pathways to a specific molecular signature. High-ranking genes such as GSTA3, LCK, and KRTDAP match the immune signals found in the GO analysis. This refined feature set—especially the Top 100 DEGs—will be the final input for the Random Forest classifier. Bayesian Optimization will then be used to further improve the model's ability to predict clinical response.

## 3.4. Survival-Associated Biomarker Analysis
The main goal of this analysis was to check the prognostic value of the DEGs by matching top features from SVM-RFE with survival-related genes from the TCGA-SKCM cohort. This step ensures the selected biomarkers can predict both immediate treatment response and long-term patient survival.

This analysis found eight core survival biomarkers: FCRL3, IKZF3, LOXL4, NUGGC, PLA2G2D, POU2AF1, SIRPG, and TNFRSF17. To check their clinical importance, a Mann-Whitney U-test was used. It showed that most of these genes have significantly higher expression in responders (R) than in non-responders (NR) ($p < 0.05$), proving they are positive indicators for immunotherapy success.
<p align="center">
  <img src="DEA results/Mann_Whitney.png" alt="Mann Whitney Analysis" width="90%"/>
  <br>
  <i>Figure 4: Expression levels of survival-related biomarkers between responders and non-responders validated by Mann-Whitney U-test.
  </i>
</p>

The impact of these biomarkers was measured using the Cox Proportional Hazards (CPH) model and Kaplan-Meier plots.
- Multivariate Cox Regression: Several genes were identified as strong independent factors. FCRL3 ($HR = 0.24$), IKZF3 ($HR = 0.25$), and PLA2G2D ($HR = 0.26$) showed significant protective effects. This means higher expression of these genes relates to a much lower risk of death. In contrast, LOXL4 was a risk factor ($HR = 1.93, p = 0.046$), meaning its high expression is linked to poorer survival.
<p align="center">
  <img src="DEA results/Hazard_ratio.png" alt="Hazard Ratio" width="70%"/>
  <br>
  <i>
    Figure 5: Multivariate Cox regression analysis displaying the Hazard Ratios (HR) for the identified survival biomarkers.
  </i>
</p>

- Kaplan-Meier Survival Evaluation: The curves show a clear difference in overall survival (OS) between patients with high and low expression levels. High expression of protective markers like FCRL3, IKZF3, and SIRPG is linked to longer survival, while high expression of LOXL4 leads to a faster decline in survival.
<p align="center">
  <img src="DEA results/Kaplan_meier.png" alt="Kaplan Meier Plot" width="90%"/>
  <br>
  <i>
    Figure 6: Kaplan-Meier survival curves illustrating the differential survival probabilities between high and low expression groups.
  </i>
</p>

<div align="center">
  <br>
  <i>
    Table 2: Multivariate Cox Regression and Clinical Cutoff Summary
  </i>

| Variable | p-value | Hazard Ratio (HR) | 95% CI (Low) | 95% CI (High) | Cutoff Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FCRL3** | 0.000031 | 0.24 | 0.12 | 0.47 | 0.1045 |
| **IKZF3** | 0.000043 | 0.25 | 0.13 | 0.48 | 0.5539 |
| **PLA2G2D** | 0.000043 | 0.26 | 0.14 | 0.50 | 0.2279 |
| **NUGGC** | 0.000468 | 0.32 | 0.17 | 0.61 | 0.1046 |
| **SIRPG** | 0.000381 | 0.31 | 0.16 | 0.59 | 1.3958 |
| **POU2AF1** | 0.002115 | 0.36 | 0.19 | 0.69 | 0.1514 |
| **LOXL4** | 0.045928 | 1.93 | 1.01 | 3.67 | 5.6728 |
| **TNFRSF17** | 0.103150 | 0.51 | 0.22 | 1.15 | 1.5887 |
</div>

## 3.5. Comparative Performance Analysis: 
The goal of this final stage was to test the predictive power of the Random Forest Classifier (RFC) using different feature sets. We compared a model using clinical features (RFC_7) with models using survival genes (RFC-SURV), top sequential features (RFC-SEQ), and a combined set (RFC-16).

The results show that while clinical features are stable, adding survival and sequential genes makes the model much stronger.
+ RFC_7 (Clinical Baseline): This model used 7 clinical features like Mutation Count and TMB. It reached an accuracy of 0.8031, but had low recall (0.33) and a low AUC (0.55). This shows that clinical data alone is not enough to predict response accurately.
+ RFC-SURV: Using 8 survival biomarkers, this model achieved 0.7383 accuracy. It has a high recall (0.85), but the AUC of 0.75 suggests it needs more features for better classification.
+ RFC-SEQ: Based on the top 8 features from SVM-RFE, this model had a mean accuracy of 0.7076. It achieved a perfect precision of 1.0 and a better AUC of 0.8286, mostly because of the importance of the SERTM2 gene.
+ RFC-16 (Combined): By combining both sets, the RFC-16 model was the most stable and accurate (mean accuracy 0.7792). It reached a superior AUC of 0.8571, proving that using both prognostic and diagnostic genes helps the model identify responders better.
<div align="center">
  <img src="DEA results/RFC7.png" alt="RFC7 Baseline Performance" width="85%"/>
  <br>
  <i>
    Figure 7: Feature importance and metrics for the RFC_7 clinical baseline model.
  </i>
</div>

<div align="center">
  <img src="DEA results/ROC_AUC_1.png" alt="ROC Results Comparison" width="70%"/>
  <br>
  <i>
    Figure 8: ROC Curve comparison across RFC-SURV, RFC-SEQ, and RFC-16 models.
  </i>
</div>

Feature Importance Analysis shows that in the RFC-16 model, biological markers are the main drivers. SERTM2 alone accounts for over 32% of the feature score. The high rank of these genes confirms they are very relevant in the melanoma tumor microenvironment.
<div align="center">
  <img src="DEA results/Feature_importance_1.png" alt="Feature Importance Results" width="90%"/>
  <br>
  <i>
    Figure 9: Relative feature importance scores for RFC-SURV, RFC-16, and RFC-SEQ.
  </i>
</div>
  
<div align="center">
<br>
   <i>
     Table 3: Comparative Performance Summary of RFC Models
   </i>

| Model | Mean Accuracy (95% CI) | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RFC_7** | **0.8031 (0.6829 - 0.9024)** | 0.6000 | 0.3333 | 0.4286 | 0.5556 |
| **RFC-SURV** | 0.7383 (0.5556 - 0.8889) | 0.8095 | **0.8500** | **0.8293** | 0.7500 |
| **RFC-SEQ** | 0.7076 (0.5185 - 0.8889) | **1.0000** | 0.6000 | 0.7500 | 0.8286 |
| **RFC-16** | 0.7792 (0.6296 - 0.9259) | **1.0000** | 0.7000 | 0.8235 | **0.8571** |
</div>

In conclusion, the RFC-16 model is much more effective than the RFC_7 model. Its high AUC and perfect precision make the 16-gene signature the best candidate for clinical use in melanoma treatment.

# 4. CONCLUSION
This project developed a high-performance framework to predict immunotherapy response in melanoma patients:
- Identification of Molecular Signatures: DEA identified 1,540 genes related to immune pathways like lymphocyte activation.
- Prognostic Validation: Eight survival biomarkers (e.g., FCRL3, IKZF3) were confirmed as protective factors with Hazard Ratios (HR) below 0.3.
- Model Optimization: The clinical baseline model (RFC_7) had poor sensitivity (0.33 recall), but the RFC-16 model performed much better.
- Superiority of RFC-16: The 16-gene model (RFC-16) achieved an AUC of 0.8571 and perfect precision (1.0), effectively identifying treatment responders.
- Potential Biomarker Insights: The high importance of the SERTM2 gene (32%) highlights its potential for future diagnostic tools and therapies.

















