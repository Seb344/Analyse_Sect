# -*- coding: utf-8 -*-
# Nouveau pdf_processor.py
# Utilise pdfplumber pour la compatibilité avec Streamlit Cloud

import pandas as pd
import numpy as np
import re
import logging
import io
import pdfplumber

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONSTANTES ---
# Mappage des colonnes (à adapter si nécessaire)
COLUMN_MAPPING = {
    'Ref enreg': 'Ref enreg',
    'Commune': 'nom_commune',
    'Adresse': 'adresse_brute',
    'Date vente': 'date_mutation',
    'Prix ( E )': 'valeur_fonciere',
    'Surface utile': 'surface_reelle_bati',
    'Nbre pièces': 'nombre_pieces_principales',
    'Année construct': 'annee_construction',
    'Surface terrain': 'surface_terrain',
    'Etage': 'etage',
    'Surface carrez': 'surface_carrez'
}

# Fonction principale de traitement PDF
def charger_et_traiter_pdf_ventes(pdf_file, pages_to_process='all'):
    """
    Traite un fichier PDF en utilisant pdfplumber pour extraire les tableaux de ventes.
    Retourne un DataFrame Pandas contenant les données extraites.
    """
    df_final_result = pd.DataFrame()
    pdf_bytes = io.BytesIO(pdf_file.getvalue())

    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            tables = []
            
            # Traite toutes les pages ou une sélection
            if pages_to_process == 'all':
                pages = pdf.pages
            else:
                pages = [pdf.pages[i-1] for i in eval(pages_to_process)]

            # Extrait les tableaux de chaque page
            for page in pages:
                # pdfplumber.extract_tables() extrait les tableaux
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    # Convertit chaque tableau en DataFrame
                    df_temp = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df_temp)

            if not tables:
                logging.warning("Aucun tableau n'a été trouvé dans les pages spécifiées.")
                return pd.DataFrame()

            # Concatène tous les DataFrames en un seul
            df_combined = pd.concat(tables, ignore_index=True)

            # Nettoyage et préparation des données
            df_combined = df_combined.rename(columns=COLUMN_MAPPING)

            if 'date_mutation' in df_combined.columns:
                df_combined['date_mutation'] = pd.to_datetime(df_combined['date_mutation'], errors='coerce')

            # Nettoyage des colonnes numériques
            for col in ['valeur_fonciere', 'surface_reelle_bati', 'surface_terrain', 'surface_carrez']:
                if col in df_combined.columns:
                    df_combined[col] = df_combined[col].astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
                    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
            
            df_final_result = df_combined.dropna(subset=['valeur_fonciere'])
            
            logging.info("Traitement PDF terminé avec succès.")
            
    except Exception as e:
        logging.error(f"Erreur lors du traitement du PDF avec pdfplumber : {e}", exc_info=True)
        return pd.DataFrame()

    return df_final_result

# --- Bloc Test (inchangé) ---
if __name__ == "__main__":
    print("--- Mode Test du module pdf_processor ---")
    # Pour tester, vous devez fournir un chemin de fichier PDF
    # test_pdf_path = r"/chemin/vers/votre/fichier.pdf"
    # Exemple d'utilisation (décommenter et modifier le chemin)
    # with open(test_pdf_path, 'rb') as f:
    #     df_result = charger_et_traiter_pdf_ventes(io.BytesIO(f.read()))
    #     if not df_result.empty:
    #         print("Résultat du test :")
    #         print(df_result.head())
    #     else:
    #         print("Le test n'a pas retourné de données.")
