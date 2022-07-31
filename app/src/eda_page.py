import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

st.set_option('deprecation.showPyplotGlobalUse', False)

file_loc1 = Path(__file__).parents[1] / '/data/ar-2010-2014-csv.csv'
file_loc2 = Path(__file__).parents[1] / '/data/ar-2015-2016-csv.csv'

file_loc1 = open(file_loc1)
file_loc2 = open(file_loc2)

def read_data(path1,path2, set_index = None):
    data1 = pd.read_csv(path1, index_col = set_index)
    data2 = pd.read_csv(path2, index_col = set_index)
    frames = [data1,data2]
    
    combined = pd.concat(frames)
    
    return combined

data = read_data(file_loc1,file_loc2)

def exclude_di(data_in):
    """
    Function to exlude DI data
    """
    data = data_in.copy()
    data = data[data['Type of treatment - IVF or DI'] != 'DI']
    return data

excluded_ = exclude_di(data)

# feature to include
feats = ['Patient Age at Treatment',
        'Total Number of Previous IVF cycles',
        'Total number of IVF pregnancies',
        'Total number of live births - conceived through IVF',
        'Type of Infertility - Female Primary',
        'Type of Infertility - Female Secondary',
        'Type of Infertility - Male Primary',
        'Type of Infertility - Male Secondary',
        'Type of Infertility -Couple Primary',
        'Type of Infertility -Couple Secondary',
        'Cause  of Infertility - Tubal disease',
        'Cause of Infertility - Ovulatory Disorder',
        'Cause of Infertility - Male Factor',
        'Cause of Infertility - Patient Unexplained',
        'Cause of Infertility - Endometriosis',
        'Cause of Infertility - Cervical factors',
        'Cause of Infertility - Female Factors',
        'Cause of Infertility - Partner Sperm Concentration',
        'Cause of Infertility -  Partner Sperm Morphology',
        'Causes of Infertility - Partner Sperm Motility',
        'Cause of Infertility -  Partner Sperm Immunological factors',
        'Stimulation used',
        'Egg Source',
        'Sperm From', 
        'Fresh Cycle', 
        'Frozen Cycle', 
        'Eggs Thawed',
        'Fresh Eggs Collected', 
        'Eggs Mixed With Partner Sperm',
        'Embryos Transfered',
        'Live Birth Occurrence']

def filtered(data_in):
    '''
    Function to select only the necessary feature
    '''
    data = data_in.copy()
    feature_list = feats
    data = data[feats]
    
    return data

filtered_ = filtered(excluded_)

def imputation(df_in):
    data = df_in.fillna(0)
    
    return data

impute = imputation(filtered_)

age_convert = {'18 - 34':0, '35-37':1, '38-39':2, '40-42':3, '43-44':4, '45-50':5}

data.drop(data[data['Patient Age at Treatment'] == '999'].index, inplace = True)

def replace_age(input_data, cats):
    """
    Function to encode age
    """
    data = input_data.copy()
    data.drop(data[data['Patient Age at Treatment'] == '999'].index, inplace = True)
    data['Patient Age at Treatment'] = data['Patient Age at Treatment'].replace(cats)
    
    return data

data = replace_age(impute, age_convert)

def to_numeric(input_data, do=True):
    '''
    Function to convert string to numerical data
    '''
    data = input_data.copy()
    
    # replace '> 50' with 51
    data['Fresh Eggs Collected'] = data['Fresh Eggs Collected'].replace(['> 50'],[51])
    data['Eggs Mixed With Partner Sperm'] = data['Eggs Mixed With Partner Sperm'].replace(['> 50'],[51])
    
    # replace '>=5' with 6
    data['Total Number of Previous IVF cycles'] = data['Total Number of Previous IVF cycles'].replace(['>=5'],[6])
    data['Total number of IVF pregnancies'] = data['Total number of IVF pregnancies'].replace(['>=5'],[6])
    
    # convert to numerical data
    data['Fresh Eggs Collected'] = pd.to_numeric(data['Fresh Eggs Collected'])
    data['Eggs Mixed With Partner Sperm'] = pd.to_numeric(data['Eggs Mixed With Partner Sperm'])
    data['Total Number of Previous IVF cycles'] = pd.to_numeric(data['Total Number of Previous IVF cycles'])
    data['Total number of IVF pregnancies'] = pd.to_numeric(data['Total number of IVF pregnancies'])
    
    return data

data = to_numeric(data)

et = data[data['Eggs Thawed'] != 0].reset_index()

def show_explore():
    st.title("Explore IVF Live-Birth Occurence")
    st.header("Human Fertilisation and Embryology Authority (HFEA)")

    col1, col2 = st.columns(2)

    st.write(
        """
    ### 1. Live Birth Occurrence
    """
    )

    plt.figure(figsize=(8,8))
    fig1, ax1 = plt.subplots()
    ax1.pie(impute['Live Birth Occurrence'].value_counts(), explode=[0,0.1],autopct='%1.1f%%',shadow=True)
    ax1.axis("equal")

    st.pyplot(fig1)

    st.caption("From the graph above we can tell that the target variable data is highly imbalance with over 75% of 0 values.")

    st.write(
        """
    ### 2. Patient at Age Treatment
    """
    )

    fig2 = plt.figure(figsize=(12,5))
    sns.set(rc={'figure.facecolor':'white'})
    sns.countplot('Patient Age at Treatment', hue='Live Birth Occurrence', data=data)

    st.pyplot(fig2)
   
    sns.factorplot('Patient Age at Treatment','Live Birth Occurrence',data=data)
    fig3 = plt.show()

    st.pyplot(fig3)

    st.caption("From this field we can tell that younger patients are more likely to have live birth. Patients under the age of 35 have the highest success rate.")

    st.write(
        """
    ### 3. Total Number of Previous IVF cycles
    """
    )

    fig4 = plt.figure(figsize=(12,5))
    sns.countplot('Total Number of Previous IVF cycles',hue='Live Birth Occurrence',data=data)

    st.pyplot(fig4)

    st.caption("Patients who've been through many IVF cycles have a smaller chance of live birth. This might have correlation with the patient's age at treatment since older patients are more likely to go through many IVF cycles.")

    st.write(
        """
    ### 4. Egg Source
    """
    )

    sns.factorplot('Egg Source','Live Birth Occurrence', data=data)
    fig5 = plt.show()

    st.pyplot(fig5)

    st.caption("Treatment with donor eggs is more likely to succeed.")

    st.write(
        """
    ### 5. Donor Sperm
    """
    )

    sns.factorplot('Sperm From','Live Birth Occurrence', data=data)
    fig6 = plt.show()

    st.pyplot(fig6)

    st.caption("Treatment with donor sperm is more likely to succeed.")

    st.write(
        """
    ### 6. Fresh or Frozen Cycle
    """
    )

    sns.factorplot('Fresh Cycle','Live Birth Occurrence', data=data)
    sns.factorplot('Frozen Cycle','Live Birth Occurrence', data=data)
    fig7 = plt.show()

    st.pyplot(fig7)

    st.caption("Treatment with frozen embryos will be more likely to succeed compared with fresh embryos.")

    st.write(
        """
    ### 7. Eggs Thawed
    """
    )

    fig8 = plt.figure(figsize=(10,6))
    plt.grid(True, alpha=0.5)
    sns.kdeplot(et.loc[et['Live Birth Occurrence'] == 0, 'Eggs Thawed'], label = 'Not Occured')
    sns.kdeplot(et.loc[et['Live Birth Occurrence'] == 1, 'Eggs Thawed'] ,  label = 'Occured')
    plt.xlabel('Eggs Thawed')
    plt.ylabel('Density')
    plt.legend()

    st.pyplot(fig8)

    st.caption("The more eggs thawed in the frozen cycle, the more likely the treatment to succeed.")

    st.write(
        """
    ### 8. Fresh Eggs Collected
    """
    )

    fec = data[data['Fresh Eggs Collected'] != 0].reset_index()

    fig9 = plt.figure(figsize=(10,6))
    plt.grid(True, alpha=0.5)
    sns.kdeplot(fec.loc[fec['Live Birth Occurrence'] == 0, 'Fresh Eggs Collected'], label = 'Not Occured')
    sns.kdeplot(fec.loc[fec['Live Birth Occurrence'] == 1, 'Fresh Eggs Collected'] ,  label = 'Occured')
    plt.xlabel('Fresh Eggs Collected')
    plt.ylabel('Density')
    plt.legend()

    st.pyplot(fig9)

    st.caption("The more fresh eggs collected, the more likely the treatment to succeed.")




