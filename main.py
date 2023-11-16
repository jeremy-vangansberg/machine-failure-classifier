import streamlit as st
import pandas as pd
import joblib

def load_model(model_path):
    pass

def make_predictions(model, data):
    pass

def main():
    st.title("Outil de prédiction des catégories")
    st.text("Fonctionnement de l'application, le modèle utilise le contenu de la colonne \n'Description' afin de prédire le sous-ensemble. Le modèle se base sur les \nincidents déjà catégorisés par des humains pour apprendre. \nJ'ai ignoré les sous-ensembles dont le nombre d'occurences est inférieur à 10.")

    uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=['xlsx'])
    
    if uploaded_file is not None:
        # Lecture du fichier Excel
        df = pd.read_excel(uploaded_file)

        model = load_model('chemin/vers/votre/modele.pkl')  

        predictions = make_predictions(model, df)
        
        df['Prédictions'] = predictions  # Ajouter les prédictions au dataframe

        # Sauvegarder le dataframe modifié en tant que nouveau fichier Excel
        output_file = 'predictions.xlsx'
        df.to_excel(output_file, index=False)

        # Permettre le téléchargement du nouveau fichier
        st.download_button(
            label="Télécharger le fichier Excel avec prédictions",
            data=output_file,
            file_name=output_file,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
