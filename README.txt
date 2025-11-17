3 FILE CODE CHÍNH:
+1.DATA_PREPROCESSING.ipynb: Xử lý và chuẩn bị 2 tập dữ liệu dữ liệu chính GSE91061 và GSE78220.
+2.PROJECT.ipynb: Thực hiện các quy trình chính bao gồm DEA->Đáp ứng mức độ miễn dịch->Gene ontology enrichment analysis->huấn luyện mô hình SVM-RFE->Xác định DEGs liên quan đến khả năng sống còn->Huấn luyện các mô hình RFC-surv, RFC-seq, RFC16.
+3.RFC7.ipynb: Huấn luyện mô hình RFC7

4 FOLDER DỮ LIỆU:
+1.melanoma: Dữ liệu gốc 8  bộ dataset (Chưa được xử lý)
+2.Preprocessed_data: Dữ liệu 2 bộ dataset GSE782220 và GSE91061 (Đã được xử lý) và dữ liệu đầu vào cho phân tích đáp ứng mức độ miễn dịch
+3.Top_genes_SVM_RFE:Chứa 3 file pickle của top 4 up-regulated genes, top 4 down-regulated genes, và top 100 genes.
+4.DEA results: Kết quả của phân tích DEA
+5.xcell_results: Các file kết quả phân tích đáp ứng mức độ miễn dịch

UTILS.PY: File source code các hàm hỗ trợ phân tích và trực quan hóa