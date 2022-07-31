from copyreg import pickle
import streamlit as st
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

pkl_path = Path(__file__).parents[1] / 'model/best_model.pkl'

model_ = open(pkl_path, 'rb')
model = pickle.load(model_)


params_predict = {'age_replace': {'18 - 34':0, '35-37':1, '38-39':2, '40-42':3, '43-44':4, '45-50':5},
                'to_dummy': ['spermFrom'],
                'to_remove': ['infertilityFemaleFactors',
                            'infertilityPartnerSpermImmuno',
                            'infertilityCouplePrimary', 
                            'infertilityMalePrimary', 
                            'frozenCycle', 'freshCycle', 
                            'numLiveBirthsIVF', 
                            'eggsMixed'],
                'to_numeric': True,
                'replace_age': True,
                'get_dummies': True,
                'replace_eggsrc': True,
                'remove_cols': True,
            }

def to_numeric(input_data, do=True):
    '''
    Function to convert string to numerical data
    '''
    data = input_data.copy()
    
    # replace '> 50' with 51
    data['freshEggsCollected'] = data['freshEggsCollected'].replace(['> 50'],[51])
    data['eggsMixed'] = data['eggsMixed'].replace(['> 50'],[51])
    
    # replace '>=5' with 6
    data['prevIVFCycles'] = data['prevIVFCycles'].replace(['>=5'],[6])
    data['numIVFPregnancies'] = data['numIVFPregnancies'].replace(['>=5'],[6])
    
    # convert to numerical data
    data['freshEggsCollected'] = pd.to_numeric(data['freshEggsCollected'])
    data['eggsMixed'] = pd.to_numeric(data['eggsMixed'])
    data['prevIVFCycles'] = pd.to_numeric(data['prevIVFCycles'])
    data['numIVFPregnancies'] = pd.to_numeric(data['numIVFPregnancies'])
    
    return data

def replace_age(input_data, cats, do=True):
    
    data = input_data.copy()
    data.drop(data[data['age'] == '999'].index, inplace = True)
    data['age'] = data['age'].replace(cats)
    
    return data

def get_dummies(input_data, col, do=True):
    
    data = input_data.copy()
    data = pd.get_dummies(data, columns=col, prefix=col)
    
    return data

def replace_eggsrc(input_data, do=True):
    
    data = input_data.copy()
    data['eggSource'] = data['eggSource'].replace(['Patient','Donor'],[0,1])
    
    return data

def remove_cols(input_data, cols, do=True):
    
    data = input_data.copy()
    data = data.drop(columns=cols)
    
    return data

def preprocess(input_data, params):
    """
    A function to execute the preprocessing steps.
    
    Args:
    - df_in(DataFrame): Input dataframe
    - params(dict): preprocessing parameters
    
    Return:
    - df(DataFrame): preprocessed data
    """
    data = input_data.copy()
    data = to_numeric(data, params['to_numeric'])
    data = replace_age(data, params['age_replace'], params['replace_age'])
    data = replace_eggsrc(data, params['replace_eggsrc'])
    data = remove_cols(data, params['to_remove'], params['remove_cols'])

    return data

def df_constructor(input):
    df = pd.DataFrame(input, index=[0])
    return df

def main_predict(data, model, params_preprocess):
    df = df_constructor(data)
    df_preprocessed = preprocess(df, params_preprocess)
    
    code2rel = {0: 'Not Occured', 1: 'Occured'}
    proba = model.predict_proba(df_preprocessed)[:,1]
    predict = 1 if proba > 0.5 else 0
    
    return code2rel[predict], proba

def show_prediction():
    st.title("Predict Live-Birth Occurence")

    st.write("""### Please input the information of the patients""")

    age = (
        "18 - 34", 
        "35-37", 
        "38-39", 
        "40-42", 
        "43-44", 
        "45-50",
    )
    
    #stimulationUsed = # Stimulation used

    eggSource = (
        "Patient",
        "Donor",
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Age & Historical Data",
                                            "Type of Infertility",
                                            "Cause of Infertility",
                                            "Sperm & Eggs",
                                            "Predict"])

    with tab1:
        age = st.selectbox("Age", age, help="Patient Age")
        
        prevIVFCycles = st.text_input("Total Number of Previous IVF cycles", "Please input >=5 if more than 5")
        numIVFPregnancies = st.text_input("Total number of IVF pregnancies", "Please input >=5 if more than 5") 

        numLiveBirthsIVF = st.number_input("Total number of live births - conceived through IVF", step=1, min_value=0)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            infertilityFemPrimary = st.selectbox("Type of Infertility - Female Primary", (0, 1,))
            infertilityFemSecondary = st.selectbox("Type of Infertility - Female Secondary", (0, 1))
            infertilityMalePrimary = st.selectbox("Type of Infertility - Male Primary", (0, 1))

        with col2:   
            infertilityMaleSecondary = st.selectbox("Type of Infertility - Male Secondary", (0, 1))
            infertilityCouplePrimary = st.selectbox("Type of Infertility -Couple Primary", (0, 1))
            infertilityCoupleSecondary = st.selectbox("Type of Infertility -Couple Secondary", (0, 1))

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            infertilityTubalDisease = st.selectbox("Cause of Infertility - Tubal disease", (0, 1))
            infertilityOvulDisorder = st.selectbox("Cause of Infertility - Ovulatory Disorder", (0, 1))
            infertilityMaleFactor = st.selectbox("Cause of Infertility - Male Factor", (0, 1))
            infertilityPatientUnexp = st.selectbox("Cause of Infertility - Patient Unexplained", (0, 1))
            infertilityEndometriosis = st.selectbox("Cause of Infertility - Endometriosis", (0, 1))
            infertilityCervicFactors = st.selectbox("Cause of Infertility - Cervical factors", (0, 1))

        with col2:
            infertilityFemaleFactors = st.selectbox("Cause of Infertility - Female Factors", (0, 1))
            infertilityPartnerSpermConc = st.selectbox("Cause of Infertility - Partner Sperm Concentration", (0, 1))
            infertilityPartnerSpermMorp = st.selectbox("Cause of Infertility -  Partner Sperm Morphology", (0, 1))
            infertilityPartnerSpermMot = st.selectbox("Causes of Infertility - Partner Sperm Motility", (0, 1))
            infertilityPartnerSpermImmuno = st.selectbox("Cause of Infertility -  Partner Sperm Immunological factors", (0,1))

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            stimulationUsed = st.selectbox("Stimulation used", (0, 1))

            eggSource = st.selectbox("Egg Source", eggSource)

            spermFromDonor = st.selectbox("Sperm From Donor", (0,1))
            spermFromPartnerandDonor = st.selectbox("Sperm From Partner & Donor", (0,1))
            spermFromNotAssigned = st.selectbox("Sperm From Not Assigned", (0,1))

            freshCycle = st.selectbox("Fresh Cycle", (0, 1))
            frozenCycle = st.selectbox("Frozen Cycle", (0, 1))

        with col2:
            eggsThawed = st.number_input("Eggs Thawed", step=1, min_value=0)

            freshEggsCollected = st.text_input("Fresh Eggs Collected", "Please input > 50 if more than 50")
            eggsMixed = st.number_input("Eggs Mixed With Partner Sperm", step=1, min_value=0)

            embryosTransfered = st.number_input("Embryos Transfered", step=1, min_value=0)
    
    with tab5:
        ok = st.button("Predict Now")
        if ok:
            X = {'age': age,
                'prevIVFCycles': prevIVFCycles,
                'numIVFPregnancies': numIVFPregnancies,                               
                'numLiveBirthsIVF': numLiveBirthsIVF,    
                'infertilityFemPrimary': infertilityFemPrimary,                         
                'infertilityFemSecondary': infertilityFemSecondary,                        
                'infertilityMalePrimary': infertilityMalePrimary,                          
                'infertilityMaleSecondary': infertilityMaleSecondary,                      
                'infertilityCouplePrimary': infertilityCouplePrimary,                        
                'infertilityCoupleSecondary': infertilityCoupleSecondary,                     
                'infertilityTubalDisease': infertilityTubalDisease,                     
                'infertilityOvulDisorder': infertilityOvulDisorder,                 
                'infertilityMaleFactor': infertilityMaleFactor,                         
                'infertilityPatientUnexp': infertilityPatientUnexp,                  
                'infertilityEndometriosis': infertilityEndometriosis,                        
                'infertilityCervicFactors': infertilityCervicFactors,                     
                'infertilityFemaleFactors':infertilityFemaleFactors,                      
                'infertilityPartnerSpermConc': infertilityPartnerSpermConc,        
                'infertilityPartnerSpermMorp': infertilityPartnerSpermMorp,          
                'infertilityPartnerSpermMot': infertilityPartnerSpermMot,              
                'infertilityPartnerSpermImmuno': infertilityPartnerSpermImmuno,   
                'stimulationUsed': stimulationUsed,                                         
                'eggSource': eggSource,                                                                                           
                'freshCycle': freshCycle,                                               
                'frozenCycle': frozenCycle,                                                
                'eggsThawed': eggsThawed,                                            
                'freshEggsCollected': freshEggsCollected,                                       
                'eggsMixed': eggsMixed,                              
                'embryosTransfered': embryosTransfered,
                'spermFromDonor': spermFromDonor,
                'spermFromPartnerandDonor': spermFromPartnerandDonor,
                'spermFromNotAssigned': spermFromNotAssigned
            }

            predict, proba = main_predict(X, model, params_predict)

            st.subheader(f"The live-birth will be {predict} with {proba[0]:.2f} probability")