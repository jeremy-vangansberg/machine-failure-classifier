import streamlit as st
import pandas as pd
import datetime
import time
from myfunctions import get_prediction, to_excel
from io import BytesIO
import xlsxwriter


st.set_page_config(
    page_title="Outil de prédiction des catégories",
    page_icon="🧊",
    layout="wide"
    )





#st.title("Outil de prédiction des catégories")
st.markdown("<h1 style='text-align: center'>Outil de prédiction des catégories</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center'>L'objectif principal de cette application est d'automatiser la prédiction des catégories pour des descriptions présentes dans un fichier Excel.</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center'>Application développée par : Jérémy VANGANSBERG, Steven DELVAL, Emad DARWICH</h5>", unsafe_allow_html=True)


# Chargez le fichier Excel
uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=['xlsx'])

if uploaded_file is not None:
    # Lecture du fichier Excel
    df = pd.read_excel(uploaded_file)

    try:
        # Sélectionnez uniquement les colonnes 'Description', 'sous ensemble'
        selected_columns = ['Description', 'sous ensemble ']
        df = df[selected_columns]
    except:
        st.warning("Vérifiez si votre fichier Excel contient les deux colonnes : 'Description' et 'sous ensemble'.", icon="⚠️")
        
    # Sélectionnez uniquement les lignes où 'sous ensemble ' est "#non catégorisé"
    # df_filtered = df[df['sous ensemble '] == "#non catégorisé"]

    # Affichez le nombre de lignes
    st.write("Nombre de lignes à prédire :", len(df))

    text_list = df['Description'].values.tolist()

    columns = st.columns((2, 1, 2))
    button = columns[1].button('Effectuer une prédiction')


    col1, col2 = st.columns(2)
    

    
    with col1:

            if button :

                # Code de l'application Streamlit
                st.markdown(""" #### API Prediction App""")

                with st.spinner("Appel API ..."):
                    # Enregistrez le temps de début
                    start_time = time.time()
                    result = get_prediction(text_list)
                    df = pd.concat([df, result], axis=1)

                    # Remplacez les valeurs de 'category' par 'à catégoriser par un humain' lorsque la 'probability' est inférieure à 0.6
                    df.loc[df['probability'] < 0.6, 'category'] = 'inférieur au seuil (0.6)'
                    df_pred = df.rename(columns={'category': 'sous ensemble estimé'})

                    # Enregistrez le temps de fin
                    end_time = time.time()
            

                    
                    



                    if df_pred  is not None:
                        st.dataframe(df_pred)
                        with col2:          
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                # Calculate the percentage
                                percentage = (len(df_pred[df_pred['sous ensemble estimé'] == 'à catégoriser par un humain']) / len(df_pred)) * 100
                                st.write(f"Temps de calcul : {round(end_time - start_time, 2)} secondes")
                                st.write(f"le pourcentage de lignes prédit comme 'à catégoriser par un humain' : {percentage}%")
                                
                                
                                df_xlsx = to_excel(df_pred)   
                                st.download_button(label='📥 Télécharger les résultats au format Excel',
                                                                data=df_xlsx ,
                                                                file_name= 'df_test.xlsx')




    
                    

                
