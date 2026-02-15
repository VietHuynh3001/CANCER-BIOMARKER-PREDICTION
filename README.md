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
- The primary objective of this analysis is to identify genes exhibiting significant expression variations between treatment responders and non-responders, thereby establishing a high-quality candidate pool for subsequent feature selection stages. By applying a stringent significance threshold of an adjusted p-value ($p_{adj}$) < 0.05 and an absolute $log_2(\text{fold change}) > 1$, the analysis successfully identified a total of `1,540 differentially expressed genes (DEGs)`. Among these, the distribution was relatively balanced, comprising `745 up-regulated genes` and `795 down-regulated genes`. The resulting Volcano Plot illustrates a robust transcriptomic signature, characterized by numerous genes achieving high statistical significance ($-log_{10}p_{adj} > 12$) and substantial magnitude of change ($log_2FC$ ranging from approximately -5 to 10). This diverse set of DEGs serves as the foundational data for the ensuing Chi-square filtering and SVM-RFE prioritization steps.

![Vocalno plot](volcano_plot.png)






