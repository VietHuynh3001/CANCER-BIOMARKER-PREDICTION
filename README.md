# 1.INTRODUCTION:
- Melanoma remains a significant global health challenge, requiring more precise and effective therapeutic interventions. Immunotherapy, particularly Immune Checkpoint Inhibitors (ICIs), has emerged as a promising strategy, leveraging immunological memory for long-term efficacy with fewer side effects than conventional chemotherapy. However, clinical success is often limited by tumor heterogeneity and the complex tumor microenvironment, which significantly impact the binding affinity between biomarkers and therapeutic antibodies.
- This project aims to identify high-potential melanoma biomarkers through an advanced machine learning pipeline. The methodology integrates Chi-square statistical filtering and Support Vector Machine-Recursive Feature Elimination (SVM-RFE) for rigorous feature selection. To maximize predictive performance and capture non-linear biological signals, XGBoost optimized via Bayesian Hyperparameter Optimization is employed. This is followed by an in-depth analysis of the TME to provide mechanistic insights into the identified biomarkers and their correlation with immune cell infiltration.

# 2. METHOD:
## 2.1. DATA RETRIEVAL AND PREPROCESSING:
Clinical and transcriptomic data of melanoma patients treated with ICIs were retrieved from cBioPortal and GEO. Patients were classified as responders or non-responders. Differential Expression Analysis (DEA) was performed using the pydeseq2 library, with significance defined as an adjusted p-value < 0.05 and |log_2FC| > 1.0 .
## 2.2. TWO-STAGE FEATURE SELECTION:
To refine the feature set, a two-stage selection process was implemented:
+ Stage 1: Chi-square Test: A statistical filter was applied to remove genes whose expression levels showed no significant association with clinical response.
+ Stage 2: SVM-RFE: A Support Vector Machine-Recursive Feature Elimination algorithm was used to rank features and select the most discriminative subset by iteratively removing low-weight genes.
## 2.3. SURVIVAL-ASSOCIATED BIOMARKER ANALYSIS
High-ranking DEGs from SVM-RFE were intersected with survival-related genes from the TCGA-SKCM cohort. The Mann-Whitney U-test was performed to compare the expression levels of these survival-related genes between responders and non-responders to ensure their association with treatment outcomes. The Cox proportional hazard model then assessed the independent impact of these genes on overall survival. Kaplan-Meier plots were generated to visualize survival probability differences between high and low expression groups.
## 2.4.GENE ONTOLOGY ENRICHMENT ANALYSIS
Functional enrichment analysis was performed using the clusterProfiler package to explore Biological Processes, Molecular Functions, and Cellular Components. A significance threshold of adjusted p-value < 0.05 was applied to identify immune-related pathways.
## 2.5.XGBOOST MODEL AND BAYESIAN OPTIMIZATION:
Predictive models were constructed using the XGBoost algorithm to capture complex non-linear signals. To prevent overfitting and maximize the Area Under the Curve (AUC), Bayesian Optimization was utilized via BayesSearchCV to tune critical hyperparameters, including learning_rate, max_depth, gamma, and scale_pos_weight.
## 2.6.TUMOR MICROENVIRONMENT ANALYSIS:
The xCell algorithm was employed to estimate infiltration scores for 64 immune and stromal cell types. This analyzed the correlation between selected biomarkers and the tumor microenvironment (TME), including CD8+ T cells, B cells, and MDSCs, to validate biological relevance.

# 3.RESULTS:
## 3.1. Differential Expression Analysis
The primary objective of this analysis is to identify genes exhibiting significant expression variations between treatment responders and non-responders, thereby establishing a high-quality candidate pool for subsequent feature selection stages. By applying a stringent significance threshold of an adjusted p-value ($p_{adj}$) < 0.05 and an absolute $log_2(\text{fold change}) > 1$, the analysis successfully identified a total of `1,540 differentially expressed genes (DEGs)`. Among these, the distribution was relatively balanced, comprising `745 up-regulated genes` and `795 down-regulated genes`. The resulting Volcano Plot illustrates a robust transcriptomic signature, characterized by numerous genes achieving high statistical significance ($-log_{10}p_{adj} > 12$) and substantial magnitude of change ($log_2FC$ ranging from approximately -5 to 10). This diverse set of DEGs serves as the foundational data for the ensuing Chi-square filtering and SVM-RFE prioritization steps.
<p align="center">
  <img src="DEA results/volcano_plot.png" width="500"/>
  <br>
  <i>
    Figure 1: Volcano Plot of Differentially Expressed Genes (DEGs) in Melanoma.
  </i>
</p>

## 3.2. Gene Ontology Enrichment Analysis
The primary objective of this analysis was to decode the biological significance of the identified `1,540` DEGs by categorizing them into three functional domains: Biological Process (BP), Cellular Component (CC), and Molecular Function (MF). By mapping these genes to established Gene Ontology terms, the study aims to ensure that the candidate features selected for the subsequent machine learning models carry relevant biological signals, particularly those associated with the mechanisms of anti-tumor immunity and immunotherapy response.

The analysis of up-regulated genes reveals a robust enrichment in pathways critical for a favorable response to immune checkpoint inhibitors. In the Biological Process domain, there is a strong focus on lymphocyte activation and immune response signaling, achieving high statistical significance with $-log_{10}FDR$ values exceeding 12.5. Regarding Cellular Components, these genes are predominantly localized to the external side of the plasma membrane and MHC protein complexes, which are essential for antigen presentation. The Molecular Function results further highlight activities such as peptide binding and cytokine receptor activity, providing evidence that the up-regulated genes are actively involved in promoting a "hot" tumor microenvironment conducive to ICI efficacy.
<p align="center">
  <img src="DEA results/GO_up.png" alt="GO Enrichment Up" width="500" height="500"/>
  <img src="DEA results/GO_down.png" alt="GO Enrichment Down" width="500" height="500"/>
  <br>
  <i>
    Figure 2: Gene Ontology (GO) dot plots illustrating the top enriched terms for up-regulated (left) and down-regulated (right) genes.
  </i>
</p>

In contrast, the down-regulated genes are primarily associated with metabolic and structural pathways that may characterize non-responsive tumor tissues. The Biological Process domain for these genes is enriched in lipid metabolism and epidermis development, while Cellular Components show significant localization in the extracellular matrix and keratin filaments. Molecular Function analysis reveals a high degree of enrichment in catalytic and structural molecule activities. These findings suggest that the down-regulated gene set reflects altered metabolic states or structural barriers that potentially hinder immune cell infiltration and treatment effectiveness.

## 3.3. Feature Selection and SVM-RFE
The primary objective of this stage was to condense the initial pool of 1,540 DEGs into a highly discriminative subset of features. By employing a two-step process—initial statistical filtering via Chi-square followed by Support Vector Machine-Recursive Feature Elimination (SVM-RFE)—the study sought to identify the most robust biomarkers capable of predicting immunotherapy response with high precision while minimizing model complexity.

The analysis of model performance demonstrates that feature reduction significantly influences predictive accuracy and stability. The SVM-RFE algorithm was used to evaluate three distinct feature sets: the Top 4 up-regulated genes, the Top 4 down-regulated genes, and a broader set of the Top 100 DEGs.
+ The Top 100 DEG model achieved the highest overall performance, with a training accuracy of 0.9047 and a validation accuracy of 0.7971. This model also yielded a superior ROC-AUC of 0.94, indicating exceptional class separation.
+ The Top 4 up-regulated model, featuring genes such as `KRTDAP, PTPRZ1, RLBP1, and SLC9A3`, showed strong generalizability with a validation accuracy of 0.7963 and a robust AUC of 0.88.
+ The Top 4 down-regulated model (including `CUX2, SIGLEC8, SPDYC, and SERTM2`) exhibited slightly lower performance, with a validation accuracy of 0.7140 and an AUC of 0.79, suggesting that while down-regulated genes contribute to the signal, up-regulated immune-related features provide more critical predictive weight.
<p align="center">
  <img src="DEA results/SVM_RFE.png" alt="SVM-RFE Performance Metrics" width="90%"/>
  <br>
  <i>
    Figure 3: SVM-RFE validation results showing Accuracy (left) and ROC-AUC curves (right) for the selected gene subsets.
  </i>
</p>

The identification of these specific "driver genes" marks a transition from broad biological pathways to a localized molecular signature. High-ranking genes such as GSTA3, LCK, and KRTDAP align with the immune-activation signals identified in the previous GO enrichment analysis. This refined feature set—particularly the high-performing Top 100 DEGs—will serve as the finalized input for the XGBoost classifier, where Bayesian Optimization will be applied to further sharpen the model's predictive boundary for clinical response.

## 3.4. Survival-Associated Biomarker Analysis
The primary objective of this analysis was to validate the prognostic value of the identified DEGs by intersecting high-ranking features from the SVM-RFE process with survival-associated genes from the TCGA-SKCM cohort. This step ensures that the selected biomarkers are not only predictive of immediate treatment response but also serve as significant indicators of long-term patient survival outcomes.

The intersection analysis yielded a refined set of eight core survival-related biomarkers: FCRL3, IKZF3, LOXL4, NUGGC, PLA2G2D, POU2AF1, SIRPG, and TNFRSF17. To verify their clinical relevance, a Mann-Whitney U-test was performed, revealing distinct expression patterns between responders (R) and non-responders (NR). For the majority of these genes, significantly higher expression levels were observed in the responder group ($p < 0.05$), reinforcing their role as positive indicators of immunotherapy efficacy.
<p align="center">
  <img src="DEA results/Mann_Whitney.png" alt="Mann Whitney Analysis" width="90%"/>
  <br>
  <i>Figure 4: Expression levels of survival-related biomarkers between responders and non-responders validated by Mann-Whitney U-test.
  </i>
</p>

The prognostic impact of these biomarkers was further quantified using the Cox Proportional Hazards (CPH) model and visualized through Kaplan-Meier plots.
- Multivariate Cox Regression: The analysis identified several genes as strong independent prognostic factors. FCRL3 ($HR = 0.24$), IKZF3 ($HR = 0.25$), and PLA2G2D ($HR = 0.26$) demonstrated the most significant protective effects, where higher expression levels correlate with a substantially reduced risk of mortality. Conversely, LOXL4 was identified as a significant risk factor ($HR = 1.93, p = 0.046$), suggesting that its up-regulation is associated with poorer survival outcomes.
<p align="center">
  <img src="DEA results/Hazard_ratio.png" alt="Hazard Ratio" width="70%"/>
  <br>
  <i>
    Figure 5: Multivariate Cox regression analysis displaying the Hazard Ratios (HR) for the identified survival biomarkers.
  </i>
</p>

- Kaplan-Meier Survival Evaluation: The survival curves illustrate a clear and significant separation in overall survival (OS) between patients with high and low expression levels of these markers. Specifically, high expression of protective biomarkers—particularly FCRL3, IKZF3, and SIRPG—is associated with extended survival probabilities, whereas high expression of LOXL4 correlates with a steeper decline in survival over time.
<p align="center">
  <img src="DEA results/Kaplan_meier.png" alt="Kaplan Meier Plot" width="90%"/>
  <br>
  <i>
    Figure 6: Kaplan-Meier survival curves illustrating the differential survival probabilities between high and low expression groups.
  </i>
</p>

***Table 2: Multivariate Cox Regression and Clinical Cutoff Summary***
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

## 3.5. Comparative Performance Analysis: 
The primary objective of this final stage was to evaluate the predictive power of the Random Forest Classifier (RFC) across different feature sets. By comparing a model based on clinical features (RFC_7) with those trained on survival-associated genes (RFC-SURV), top sequential features (RFC-SEQ), and a combined set (RFC-16), the study aimed to identify the most parsimonious yet accurate molecular signature for predicting immunotherapy response.

The performance analysis indicates that while clinical features provide a stable baseline, integrating survival-related and discriminative sequential genes yields significantly more robust results.
+ RFC_7 (Clinical Baseline): This model utilized 7 clinical features, including Mutation Count, TMB, and Fraction Genome Altered. It achieved a mean bootstrap accuracy of 0.8031 (95% CI: 0.6829-0.9024). However, despite the high accuracy, it showed a limited recall of 0.3333, precision of 0.60, and an AUC of 0.5556, reflecting the difficulty of predicting response using only non-transcriptomic data.
+ RFC-SURV: Utilizing the 8 survival-related biomarkers, this model achieved a mean bootstrap accuracy of 0.7383 (95% CI: 0.5556-0.8889). While it maintains a high recall (0.85), its AUC of 0.75 suggests that survival signals alone, though prognostic, may require additional discriminative features for optimal classification.
+ RFC-SEQ: Based on the top 8 sequential features from SVM-RFE, this model showed a mean accuracy of 0.7076 (95% CI: 0.5185-0.8889). Notably, it achieved a perfect precision of 1.0, with an improved AUC of 0.8286, driven largely by the heavy feature importance of SERTM2.
+ RFC-16 (Combined): By combining both feature sets, the RFC-16 model reached the highest stability and accuracy, with a mean bootstrap estimate of 0.7792 (95% CI: 0.6296-0.9259). This model achieved a superior AUC of 0.8571, demonstrating that the synergy between prognostic (survival) and diagnostic (sequential) genes enhances the model's ability to distinguish responders.

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

Feature Importance Analysis across the models highlights a shift in predictive drivers. In RFC_7, the most influential factors were Mutation Count (0.1895) and TMB (0.1740). In contrast, the RFC-16 model is driven by biological markers where SERTM2 accounts for over 32% of the total feature score, followed by SLC9A3 and RLBP1. The consistent high ranking of these genes underscores their biological relevance in the melanoma tumor microenvironment.
<div align="center">
  <img src="DEA results/Feature_importance_1.png" alt="Feature Importance Results" width="90%"/>
  <br>
  <i>
    Figure 9: Relative feature importance scores for RFC-SURV, RFC-16, and RFC-SEQ.
  </i>
  
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
<br>
</div>

In conclusion, while the RFC_7 model offers high overall accuracy, the RFC-16 model provides a much more effective balance of precision and sensitivity. The superior AUC and perfect precision on test sets suggest that this 16-gene signature is the most powerful candidate for clinical decision-making in melanoma immunotherapy.











