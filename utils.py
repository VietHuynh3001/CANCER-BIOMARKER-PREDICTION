# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder
import mygene
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import auc,f1_score,accuracy_score,recall_score,precision_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder


# HÀM XỬ LÝ FILE DATA TỪ cBIOPORTAL
def precocessing_dataset_from_cBioPortal(name:str,list_selected_genes:list)->pd.DataFrame:
    # ĐỌC VÀ SỬ LÝ FILE CLINICAL PATIENTS
    df_clinical_patients=pd.read_csv(f'melanoma/{name}/data_clinical_patient.txt',
                                     delimiter='\t',
                                     header=4,
                                     index_col=0)
    df_clinical_patients=df_clinical_patients[['OS_STATUS','OS_MONTHS','SEX','AGE']]
     # TRUY XUẤT GENE ID
    mg=mygene.MyGeneInfo()
    dict_genes={int(gene['_id']) if gene['_id'].isdigit() else gene['_id']:gene['query'] \
                for gene in mg.querymany(list_selected_genes,species='human',scopes='symbol',fields='entrezgene')}
    # ĐOC VÀ SỬ LÝ FILE CNA
    df_cna=pd.read_csv(f'melanoma/{name}/data_mrna_seq_rpkm.txt',
                       delimiter='\t',index_col=0)
    df_cna=df_cna[df_cna.index.isin(dict_genes.keys())]
    df_cna.index=df_cna.index.map(dict_genes)
    df_cna=df_cna.T
    # MERGE 2 FILE 
    final_df=df_clinical_patients.merge(df_cna,left_index=True,right_index=True,how='inner')
    # ENCODE CATEGORICAL VARIABLES 
    encoder=LabelEncoder()
    final_df['OS_STATUS_ENCODED']=encoder.fit_transform(final_df['OS_STATUS']) # OVERALL SURVIVAL
    final_df['SEX_ENCODED']=encoder.fit_transform(final_df['SEX']) # SEX 
    final_df.drop(columns=['OS_STATUS','SEX'],inplace=True)
    return final_df

# COX REGRESSION CHO NUMERICAL VARIABLE 
def cox_regression_model_univariate(df_dataset:pd.DataFrame,variable:str,n:int)->(float,int):
    # XÁC ĐỊNH QUANTILE 1 VÀ QUANTILE 3 MỖI BIẾN
    q1,q3=df_dataset[variable].quantile([0.25,0.75])
    # TẠO 1 LIST CUTOFF CÓ N GIÁ TRỊ CHẠY TỪ Q1 ĐẾN Q3
    list_cutoff=np.linspace(q1,q3,n)
    # XÁC ĐỊNH CUTOFF VỚI P-VALUE THẤP NHẤT
    p_value_final=1
    cutoff_final=None
    HR_lower_final=None
    HR_upper_final=None
    HR_final=None
    for cutoff in list_cutoff:
        df_temp=df_dataset.copy()
        df_temp[f'{variable}_ENCODED']=(df_temp[variable]>cutoff).astype('bool')
        df_temp=df_temp[[f'{variable}_ENCODED','OS_STATUS_ENCODED','OS_MONTHS']]
        # ĐỊNH NGHĨA MODEL
        cox_reg=CoxPHFitter()
        # FIT MODEL
        cox_reg.fit(df=df_temp,event_col='OS_STATUS_ENCODED',duration_col='OS_MONTHS')
        # KẾT QUẢ
        p_value=cox_reg.summary.loc[f'{variable}_ENCODED','p']
        HR=round(cox_reg.summary.loc[f'{variable}_ENCODED','exp(coef)'],2)
        HR_upper=round(cox_reg.summary.loc[f'{variable}_ENCODED','exp(coef) upper 95%'],2)
        HR_lower=round(cox_reg.summary.loc[f'{variable}_ENCODED','exp(coef) lower 95%'],2)
        if p_value<p_value_final:
            p_value_final=p_value
            cutoff_final=cutoff
            HR_lower_final=HR_lower
            HR_upper_final=HR_upper
            HR_final=HR
    return p_value_final,cutoff_final,HR_final,HR_lower_final,HR_upper_final

# CHUYỂN LIST GENE ID HOẶC LIST GENE SYMBOL THÀNH DICT GENE ID:SYMBOL
def gene_dict_id_symbol(list_seleceted_genes:list)->dict:
    # GỌI ĐỐI TƯỢNG
    mg=mygene.MyGeneInfo()
    # NẾU LIST LÀ INT THÌ CHUYỂN SANG STR
    list_seleceted_genes=list(map(str,list_seleceted_genes))
    # TRƯỜNG HỢP CÁC ITEMS TRONG LIST LÀ DẠNG DIGIT (GENE ID)
    if all(list(map(lambda x:x.isdigit(),list_seleceted_genes))):
        dict_gene={gene.get('query'):gene.get('symbol') for gene in mg.getgenes(list_seleceted_genes,species='human',fields='symbol')}
    else:
        dict_gene={gene.get('_id'):gene.get('query') for gene in mg.querymany(list_seleceted_genes,fields='entrezgene',scopes='symbol',species='human')}
    return dict_gene

# LỌC NHỮNG GENE MỤC TIÊU CÓ TRONG DATAFRAME
def gene_filtered_dataframe(df:pd.DataFrame,dict_seleceted_gene:dict)->pd.DataFrame:
    list_gene_in_df=df.columns
    # TRƯỜNG HỢP DATAFRAME CÓ CỘT LÀ GENE ID: DIGIT
    if all(list(map(lambda x:x.isdigit(),list_gene_in_df))):
        df=df.loc[::,df.columns.isin(dict_seleceted_gene.keys())]
        df.rename(columns=dict_seleceted_gene,inplace=True)
    # TRƯỜNG HỢP DATAFRAME CÓ CỘT LÀ GENE SYMBOL
    else:
        df=df.loc[::,df.columns.isin(dict_seleceted_gene.values())]
    return df
# CROSS VALIDATION BẰNG RANDOM FOREST
def cross_validation_by_RFC(n_fold,X_train,y_train,scoring,seed):
    # Định nghĩa model
    RFC=RandomForestClassifier(random_state=seed)
    # Kfold
    kfold=KFold(n_splits=n_fold,random_state=seed,shuffle=True)
    # CROSS-VALIDATION
    result=cross_val_score(estimator=RFC,
                       X=X_train,
                       y=y_train,
                       scoring=scoring,
                       cv=kfold)
    return result
# TRAIN RANDOM FOREST MODEL
def training_RFC(X_train,y_train,X_test,y_test,n):
    list_score=[]
    list_accuracy_train=[]
    list_precision=[]
    list_recall=[]
    list_f1=[]
    list_accuracy_test=[]
    list_feature_importance=[]
    for i in range(n):
        # ĐỊNH NGHĨA MODEL
        RFC=RandomForestClassifier(random_state=i)
        # FIT DỮ LIỆU
        RFC.fit(X=X_train,y=y_train)
        # PREDICT
        pred=RFC.predict(X_test)
        # RESULT
        accuracy_test=accuracy_score(y_true=y_test,y_pred=pred)
        accuracy_train=RFC.score(X=X_train,y=y_train)
        recall=recall_score(y_true=y_test,y_pred=pred)
        precison=precision_score(y_true=y_test,y_pred=pred)
        f1=f1_score(y_true=y_test,y_pred=pred)
        
        list_score.append({
            'accuracy_test':accuracy_test,
            'accuracy_train':accuracy_train,
            'recall':recall,
            'precison':precison,
            'f1':f1
        })
        # FEATURE IMPORTANCE
        feature_score=RFC.feature_importances_
        list_feature_importance.append(feature_score)
    list_feature_name=RFC.feature_names_in_
    df_score=pd.DataFrame(list_score)
    df_feature_importance=pd.DataFrame(list_feature_importance,columns=list_feature_name)
    return df_score,df_feature_importance

# TÍNH TOÁN KẾT QUẢ AVERAGE VÀ STARDARD DEVIATION
def caculate_result(df):
    df_result=pd.merge(left=df.mean().rename('mean'),
                   right=df.std().rename('std'),
                   left_index=True,
                   right_index=True,how='inner')
    df_result.sort_values(by='mean',inplace=True,ascending=True)
    return df_result

# MODEL COX REGRESSION CHO BIBARY VARIABLE
def cox_regression_model(df_dataset:pd.DataFrame,variable:str)->pd.DataFrame:
    # CHUẨN BỊ
    df_temp=df_dataset[[variable,'OS_STATUS_ENCODED','OS_MONTHS']]
    # ĐỊNH NGHĨA MODEL
    cox_reg=CoxPHFitter()
    # FIT MODEL
    cox_reg.fit(df=df_temp,event_col='OS_STATUS_ENCODED',duration_col='OS_MONTHS')
    # KẾT QUẢ
    p_value=cox_reg.summary.loc[variable,'p']
    HR=round(cox_reg.summary.loc[variable,'exp(coef)'],2)
    HR_upper=round(cox_reg.summary.loc[variable,'exp(coef) upper 95%'],2)
    HR_lower=round(cox_reg.summary.loc[variable,'exp(coef) lower 95%'],2)
    return p_value,HR,HR_lower,HR_upper

# TRƯC QUAN HÓA FEATURE IMPORTANCE
def feature_importance_graph(df:pd.DataFrame,title_graph:str)->plt.Figure:
    plt.figure(figsize=(10,5))
    plt.title(title_graph.upper(),weight=800)
    plt.barh(y=df.index,
             width=df['mean'],
             color='tomato')
    plt.errorbar(x=df['mean'],
                 y=df.index,
                 xerr=df['std'],
                 color='black',
                 linestyle='none')
    plt.xlabel('Gene symbol',weight=800)
    plt.ylabel('Feature importance score',weight=800)

# TRỰC QUAN HÓA KẾT QUẢ DEG
def vocanol_plot(df_DEA:pd.DataFrame)->plt.Figure:
    # TRỰC QUAN HÓA KẾT QUẢ
    plt.figure(figsize=(10,10))
    plt.title('TRỰC QUAN HÓA KẾT QUẢ DEG',weight=900)
    # CÁC GIÁ TRỊ TRỤC X
    x_value=df_DEA[df_DEA['padj']<0.05]['log2FoldChange']
    # CÁC GIÁ TRỊ TRỤC Y
    y_value=-np.log10(df_DEA[df_DEA['padj']<0.05]['padj'])
    # UP-REGULATED GENES
    plt.scatter(x=x_value[x_value>0.5],
                y=y_value[x_value>0.5],
                color='Darkviolet',
                label='Up-regulated genes',s=5.4)
    # DOWN-REGULATED GENES
    plt.scatter(x=x_value[x_value<-0.5],
                y=y_value[x_value<-0.5],
                color='Tomato',
                label='Down-regulated genes',s=5.4)
    # GENE BÌNH THƯỜNG
    plt.scatter(x=x_value[np.abs(x_value)<=0.5],
                y=y_value[np.abs(x_value)<=0.5],
                color='Lightgreen',
                label='No',s=5.4)
    plt.vlines(x=0.5,ymin=y_value.min(),ymax=y_value.max()+1,colors='black',linewidth=0.8)
    plt.vlines(x=-0.5,ymin=y_value.min(),ymax=y_value.max()+1,colors='black',linewidth=0.8)
    plt.ylim([y_value.min(),y_value.max()+1])
    plt.xlim([-14,14])
    plt.xlabel('log2FoldChange')
    plt.ylabel('-log10(padj)')
    plt.legend(title='DEG')