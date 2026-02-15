# 1.INTRODUCTION:
Breast cancer have been one the most severe disease, posing serious problem to public health. Thus, an effective treatment on breast cancer have been necessary than ever. Immunotherapy have been considered the most potential treatment due to lack of side effects and prolonged effeciency of immunological memory. However, the effeciency of the treatment depends on the affinity between biomarker and antibody, which is changed due to tumor heterogenity caused by tumor microenviromnet. This project aim to identify the potential breast cancer biomarker using machine learning and stastical methods and perform further tumor microenvironment analysis 
# 1. SOURCE CODE
+1.DATA_PREPROCESSING.ipynb: Handles data cleaning and preprocessing for the two primary datasets GSE91061 and GSE78220.
+2.PROJECT.ipynb: The main analysis pipeline. This notebook executes:
   Differential Expression Analysis (DEA).
   Immune analysis.
   Gene Ontology enrichment analysis.
   Feature selection using SVM-RFE.
   Identification of survival-associated DEGs.
   Training of RFC models (RFC-surv, RFC-seq, RFC16).
+3.RFC7.ipynb: Training the RFC7 model.
+4.UTILS.py: Contains helper functions for data analysis and visualization used throughout the project.

2. DATA & OUTPUT
+1.melanoma/: Contains the 8 original raw datasets.
+2.Preprocessed_data/: Processed data for GSE78220 and GSE91061, along with input files for immune response analysis.
+3.Top_genes_SVM_RFE/: Stores pickle files for the top 4 up-regulated, top 4 down-regulated, and top 100 genes.
+4.DEA_results/: Output files generated from Differential Expression Analysis.
+5.xcell_results/: Results from the immune response analysis (xCell).
<b>2.RESULTS:</b>


