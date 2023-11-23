import requests
import pandas as pd
import time


def get_prediction(user_input):
    """
    Fait une requête à l'API de prédiction.

    Parameters:
        user_input (str): La description pour laquelle la prédiction est demandée.

    Returns:
        pd.DataFrame: Un DataFrame contenant les résultats de la prédiction.
    """

    # Définissez l'URL de l'API
    api_endpoint = "http://20.19.203.222/predict"

    # Préparez les données dans le format requis
    payload = {
        "description": user_input
    }

    # Faites une requête POST à l'API
    response = requests.post(api_endpoint, json=payload)

    # Vérifiez si la requête a réussi (code d'état 200)
    if response.status_code == 200:
        # Extrait le résultat de la prédiction
        result = response.json()
        # Retourne le résultat sous forme de DataFrame
        return pd.DataFrame(result)
    else:
        # Affiche une erreur si la requête n'a pas réussi
        print(f"Échec de la prédiction. Code d'état : {response.status_code}")
        return None