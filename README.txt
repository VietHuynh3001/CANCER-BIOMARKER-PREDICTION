
NOTE: THIS PROJECT IS THE REPLICATION WORK OF "Ahmed, Yaman B et al. “Genomic and Transcriptomic 
Predictors of Response to Immune Checkpoint Inhibitors in Melanoma Patients: A Machine Learning 
Approach.” Cancers vol. 14,22 5605. 15 Nov. 2022, doi:10.3390/cancers14225605". THE PROJECT SERVES AS A HANDS-ON 
EXERCISE TO DEEPEN MY KNOWLEDGE IN BIOINFORMATICS AND MACHINE LEARNING APPLICATIONS IN ONCOLOGY.

1. SOURCE CODE
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