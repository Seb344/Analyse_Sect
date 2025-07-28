# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
import numpy as np
import geopy.distance
import plotly.io as pio 
import io
import zipfile
import datetime 
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter # Pour respecter les limites Nominatim



# --- Définition des utilisateurs autorisés pour l'authentification ---
# N'oublie pas de personnaliser ces valeurs !
AUTHORIZED_USERS = {
    "utilisateur1": "motdepasse123",
    "admin": "supersecurite",
    "demo": "test1234"
}

# --- La fonction d'authentification ---
def check_password():
    """
    Fonction pour gérer la connexion de l'utilisateur.
    Elle affiche un formulaire de connexion et vérifie les identifiants.
    Retourne True si l'utilisateur est authentifié, False sinon.
    """
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    if st.session_state["authentication_status"]:
        return True

    st.title("Accès à l'application")
    st.write("Veuillez vous connecter pour continuer.")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
                st.session_state["authentication_status"] = True
                st.success(f"Bienvenue, {username} !")
                st.rerun()
                return True
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")
                st.session_state["authentication_status"] = False
                return False
    return False





# --- AJOUT: Définition de prepare_hist_data ICI ---
# (Note: Il est préférable de la définir globalement une seule fois)
def prepare_hist_data(df_source, label):
    cols_to_keep_for_hist = ['prix_m2', 'valeur_fonciere']
    if df_source is None or df_source.empty or not all(col in df_source.columns for col in cols_to_keep_for_hist):
        logging.warning(f"DataFrame source invalide ou colonnes manquantes pour '{label}' dans prepare_hist_data.")
        return None
    temp_df = df_source[cols_to_keep_for_hist].copy()
    temp_df['prix_m2'] = pd.to_numeric(temp_df['prix_m2'], errors='coerce')
    temp_df['valeur_fonciere'] = pd.to_numeric(temp_df['valeur_fonciere'], errors='coerce')
    temp_df.dropna(subset=cols_to_keep_for_hist, how='any', inplace=True)
    temp_df = temp_df[(temp_df['prix_m2'] > 0) & (temp_df['valeur_fonciere'] > 0)]
    if not temp_df.empty:
        temp_df['Source'] = label
        return temp_df
    else:
        logging.warning(f"Aucune donnée valide après filtrage pour '{label}' dans prepare_hist_data.")
        return None
# --- FIN Définition ---


# --- Fonction Helper pour calculer les stats comparatives ---
def calculate_comparative_stats(df, label):
    """
    Calcule les statistiques clés pour le tableau comparatif de l'onglet 3.
    Retourne un dictionnaire avec les valeurs formatées ET la moyenne numérique.
    """
    # Initialisation avec NaN/N/A pour les valeurs
    stats = {
        'Source': label,
        'Nb Biens': 0,
        'Prix/m² Moyen': 'N/A',
        'Prix/m² Median': 'N/A',
        'Prix/m² Min': 'N/A',
        'Prix/m² Max': 'N/A',
        'Prix/m² Moyen Num': np.nan # Pour calcul de différence %
        # 'Prix/m² Median Num': np.nan # Optionnel: ajouter si besoin
    }
    if df is None or df.empty:
        logging.warning(f"DataFrame vide ou None pour '{label}' dans calculate_comparative_stats.")
        return stats # Retourner stats initialisées si DF vide

    stats['Nb Biens'] = len(df) # Compte total basé sur le DF filtré d'origine

    if 'prix_m2' in df.columns:
        # Pour les stats prix/m², on filtre les > 0 et non-NaN/inf
        valid_prices = pd.to_numeric(df['prix_m2'], errors='coerce').dropna()
        valid_prices = valid_prices[valid_prices > 0]
        valid_prices = valid_prices[np.isfinite(valid_prices)] # Exclure infinis

        if not valid_prices.empty:
            mean_val = valid_prices.mean()
            median_val = valid_prices.median()
            min_val = valid_prices.min()
            max_val = valid_prices.max()

            # Fonction interne de formatage robuste
            def format_stat_comp(value):
                 if pd.notna(value) and np.isfinite(value):
                     try: return f"{int(round(value)):,} €/m²".replace(',', ' ')
                     except: return str(value) + " €/m²" # Fallback
                 return 'N/A'

            # Mise à jour du dictionnaire avec valeurs formatées et numériques
            stats.update({
                'Prix/m² Moyen': format_stat_comp(mean_val),
                'Prix/m² Median': format_stat_comp(median_val),
                'Prix/m² Min': format_stat_comp(min_val),
                'Prix/m² Max': format_stat_comp(max_val),
                'Prix/m² Moyen Num': mean_val # Stocker la valeur numérique
                # 'Prix/m² Median Num': median_val # Décommenter si besoin
            })
        else:
            logging.warning(f"Aucun prix/m² valide trouvé pour '{label}'.")
    else:
        logging.warning(f"Colonne 'prix_m2' manquante pour '{label}'.")

    return stats
# --- Fin Fonction Helper ---





# --- AJOUT: Importer la fonction depuis le fichier séparé ---
try:
    # Essayer d'importer la fonction depuis le fichier pdf_processor.py
    from pdf_processor import charger_et_traiter_pdf_ventes
    # Si l'import réussit, marquer la fonctionnalité comme disponible
    PDF_PROCESSOR_AVAILABLE = True
    logging.info("Module pdf_processor chargé avec succès.")

except ImportError:
    # Si l'import échoue (fichier non trouvé, etc.)
    PDF_PROCESSOR_AVAILABLE = False
    logging.error("ERREUR: Impossible d'importer 'pdf_processor.py'. Assurez-vous qu'il est dans le même répertoire.")

    # --- Début de la fonction factice (indentée sous 'except') ---
    # Définir une fonction qui porte le même nom pour éviter des erreurs 'NameError'
    # si le reste du code essaie d'appeler la fonction importée.
    def charger_et_traiter_pdf_ventes(*args, **kwargs):
        # À l'intérieur de la fonction factice (indentation supplémentaire)
        # Afficher une erreur à l'utilisateur indiquant que ça ne fonctionne pas
        st.error("Fonctionnalité de traitement PDF non disponible (Erreur d'importation du module).")
        # Retourner un DataFrame vide pour correspondre au type de retour attendu
        return pd.DataFrame()
    # --- Fin de la fonction factice ---

# --- FIN AJOUT ---

# --- AJOUT: Importations pour Seaborn/Matplotlib ---
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_MPL_AVAILABLE = True
    logging.info("Modules Seaborn et Matplotlib chargés.")
except ImportError:
    SEABORN_MPL_AVAILABLE = False
    logging.warning("Modules Seaborn ou Matplotlib non trouvés. Graphe KDE non disponible.")
# --- FIN AJOUT ---


# Configuration du logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Début: Fonctions (inchangées) ---

def load_data(file_path_or_stream, target_directory=None):
    # ... (Code complet et CORRECT de load_data - INCHANGÉ) ...
    logging.info(f"Fonction load_data appelée. Target dir: {target_directory}")
    df_selected = pd.DataFrame()
    df_final = pd.DataFrame()
    try:
        if isinstance(file_path_or_stream, str): df = pd.read_csv(file_path_or_stream, sep=',', encoding='utf-8', dtype=str)
        else: df = pd.read_csv(file_path_or_stream, sep=',', encoding='utf-8', dtype=str)
        
        # --- Nettoyage Initial NaN
        logging.info("Nettoyage initial DVF..."); df.fillna('', inplace=True)

        rows_before_drop = len(df) # Garder trace du nb initial
        logging.info(f"Nombre de lignes lues: {rows_before_drop}")

        # --- NOUVEAU: Supprimer les doublons STRICTEMENT identiques ---
        logging.info("Suppression des doublons strictement identiques...")
        df.drop_duplicates(inplace=True) # <--- LIGNE AJOUTÉE IMPORTANTE
        rows_after_drop = len(df)
        duplicates_removed_count = rows_before_drop - rows_after_drop
        if duplicates_removed_count > 0:
            logging.info(f"{duplicates_removed_count} doublons stricts supprimés.")
            st.sidebar.info(f"{duplicates_removed_count} lignes strictement identiques supprimées du fichier DVF.") # Informer l'utilisateur
        else:
            logging.info("Aucun doublon strict trouvé.")
        df.reset_index(drop=True, inplace=True) # Réinitialiser l'index après suppression
        # --- FIN NOUVEAU ---


        logging.info("Nettoyage et conversion DVF...")
        numeric_cols = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain', 'longitude', 'latitude']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        cols_to_fill_zero = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain']
        for col in cols_to_fill_zero: df[col] = df[col].fillna(0)
        df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce')
        df['annee'] = df['date_mutation'].dt.year.fillna(0).astype(int)
        df['prix_m2'] = 0.0
        mask = (pd.to_numeric(df['surface_reelle_bati'], errors='coerce').fillna(0) > 0) & (pd.to_numeric(df['valeur_fonciere'], errors='coerce').fillna(0) > 0)
        valeur_fonciere_num = pd.to_numeric(df.loc[mask, 'valeur_fonciere'], errors='coerce').fillna(0)
        surface_bati_num = pd.to_numeric(df.loc[mask, 'surface_reelle_bati'], errors='coerce').fillna(1)
        df.loc[mask, 'prix_m2'] = valeur_fonciere_num / surface_bati_num
        df.replace([np.inf, -np.inf], 0, inplace=True); df['prix_m2'] = df['prix_m2'].fillna(0)
        for col in ['adresse_numero', 'adresse_nom_voie', 'code_postal', 'nom_commune']:
             if col not in df.columns: df[col] = ''
             else: df[col] = df[col].astype(str).fillna('')
        df['adresse_Maps'] = df['adresse_numero'] + ' ' + df['adresse_nom_voie'] + ' ' + df['code_postal'] + ' ' + df['nom_commune']
        logging.info("Détection doublons DVF...")
        if 'id_mutation' in df.columns and 'type_local' in df.columns:
            df['doublon'] = 1; df['id_mutation'] = df['id_mutation'].astype(str)
            grouped = df.groupby('id_mutation')['type_local'].apply(list).reset_index(name='type_local_list')
            def is_valid(types):
                app_count = types.count('Appartement'); mai_count = types.count('Maison'); loc_count = sum(1 for tl in types if isinstance(tl, str) and tl.startswith('Local'))
                return (app_count == 1 and mai_count == 0 and loc_count == 0) or (mai_count == 1 and app_count == 0 and loc_count == 0)
            valid_ids = grouped[grouped['type_local_list'].apply(is_valid)]['id_mutation']
            mask_valid = (df['id_mutation'].isin(valid_ids)) & (df['type_local'].isin(['Appartement', 'Maison']))
            df.loc[mask_valid, 'doublon'] = 0
        else: logging.warning("Filtre 'doublon' DVF non appliqué."); df['doublon'] = 0
        all_required_columns_app = ['date_mutation', 'annee', 'nom_commune', 'section_prefixe','valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales','prix_m2', 'surface_terrain', 'type_local','longitude', 'latitude', 'adresse_Maps','doublon', 'id_mutation']
        cols_to_select = [col for col in all_required_columns_app if col in df.columns]
        df_selected = df[cols_to_select].copy()
        if 'doublon' in df_selected.columns:
            df_final = df_selected[df_selected['doublon'] == 0].copy()
            if 'doublon' in df_final.columns: df_final.drop('doublon', axis=1, inplace=True)
        else:
             df_final = df_selected.copy()
             if 'doublon' in df_final.columns: df_final.drop('doublon', axis=1, inplace=True)
        # Sauvegarde fichier DVF
        if target_directory and hasattr(file_path_or_stream, 'name'):
             try:
                base_name = os.path.splitext(os.path.basename(file_path_or_stream.name))[0]
                f_complet = os.path.join(target_directory, base_name + '_complet_nettoye.csv')
                f_filtered = os.path.join(target_directory, base_name + '_filtre_doublons.csv')
                if not df_selected.empty: df_selected.to_csv(f_complet, index=False, encoding='utf-8-sig', sep=','); st.sidebar.success(f"DVF Complet sauvegardé:\n...{os.path.basename(f_complet)}")
                else: st.sidebar.warning("Sauvegarde DVF complet annulée (vide).")
                if not df_final.empty: df_final.to_csv(f_filtered, index=False, encoding='utf-8-sig', sep=','); st.sidebar.success(f"DVF Filtré sauvegardé:\n...{os.path.basename(f_filtered)}")
                else: st.sidebar.info("Aucune donnée DVF à sauvegarder (après filtre).")
             except Exception as save_error: st.sidebar.error(f"Erreur sauvegarde DVF: {save_error}"); logging.error(f"Erreur sauvegarde DVF: {save_error}", exc_info=True)
        elif target_directory: logging.warning("Sauvegarde DVF annulée (nom fichier manquant?).")
        logging.info(f"load_data (DVF): {len(df_final)} lignes retournées.")
        return df_final
    except Exception as e: logging.error(f"Erreur DANS load_data (DVF): {str(e)}", exc_info=True); raise e


def clean_data(df):
    df = df.copy(); logging.info("Nettoyage final..."); df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    cols_to_int = ['annee', 'surface_reelle_bati', 'nombre_pieces_principales', 'valeur_fonciere', 'prix_m2', 'surface_terrain']
    for col in cols_to_int:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    float_cols = ['longitude', 'latitude']
    for col in float_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns: raise ValueError(f"Colonnes manquantes : {', '.join(missing_columns)}")

required_columns_app = ['annee', 'nom_commune', 'section_prefixe','valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales','prix_m2', 'surface_terrain', 'type_local']

def create_validation_dropdown(df, bins, labels):
    df_copy = df.copy(); df_copy['surface_reelle_bati'] = pd.to_numeric(df_copy['surface_reelle_bati'], errors='coerce').fillna(0); df_copy['prix_m2'] = pd.to_numeric(df_copy['prix_m2'], errors='coerce').fillna(0)
    df_copy.dropna(subset=['surface_reelle_bati'], inplace=True); df_copy = df_copy[np.isfinite(df_copy['surface_reelle_bati'])]
    if df_copy.empty: return {label: [] for label in labels}
    try: df_copy['surface_bin'] = pd.cut(df_copy['surface_reelle_bati'], bins=bins, labels=labels, right=False)
    except ValueError as e_cut: logging.error(f"Erreur pd.cut: {e_cut}"); return {label: [] for label in labels}
    validation_data = {}
    for label in labels:
        if label in df_copy['surface_bin'].cat.categories:
             bin_data = df_copy[df_copy['surface_bin'] == label]['prix_m2'].unique()
             valid_prices = sorted([int(price) for price in bin_data if pd.notna(price) and isinstance(price, (int, float, np.number)) and np.isfinite(price)])
             validation_data[label] = valid_prices
        else: validation_data[label] = []
    return validation_data

def calculate_validated_stats(validation_data, session_state, key_prefix):
    validated_rows = [];
    for bin_label, prices in validation_data.items():
        widget_key = f"{key_prefix}_multiselect_{bin_label}"; selected_prices_for_bin = session_state.get(widget_key, [])
        for price in selected_prices_for_bin:
             if isinstance(price, (int, float, np.number)): validated_rows.append({'Tranche': bin_label, 'Prix m²': price})
    if not validated_rows: return pd.DataFrame(columns=['Tranche', 'Prix_moyen_m2', 'Prix_median_m2', 'Nombre_de_biens'])
    validated_data = pd.DataFrame(validated_rows); validated_data['Prix m²'] = pd.to_numeric(validated_data['Prix m²'], errors='coerce'); validated_data.dropna(subset=['Prix m²'], inplace=True)
    if validated_data.empty: return pd.DataFrame(columns=['Tranche', 'Prix_moyen_m2', 'Prix_median_m2', 'Nombre_de_biens'])
    stats = validated_data.groupby('Tranche').agg(Prix_moyen_m2=('Prix m²', lambda x: int(round(x.mean())) if not x.empty else 0), Prix_median_m2=('Prix m²', lambda x: int(round(x.median())) if not x.empty else 0), Nombre_de_biens=('Prix m²', 'count')).reset_index()
    return stats

def highlight_stats_table(data):
    df = data.copy(); style_matrix = pd.DataFrame('', index=df.index, columns=df.columns)
    if len(df) >= 9:
        colors = [('#000000', '#FFFFFF', '#d9d9d9', '#000000'), ('#6fa8dc', '#FFFFFF', '#cfe2f3', '#000000'),('#6fa8dc', '#FFFFFF', '#cfe2f3', '#000000'), ('#93c47d', '#FFFFFF', '#d9ead3', '#000000'),('#93c47d', '#FFFFFF', '#d9ead3', '#000000'), ('#ff0000', '#FFFFFF', '#f4cccc', '#000000'),('#ff0000', '#FFFFFF', '#f4cccc', '#000000'), ('#f9cb9c', '#000000', '#ffe5d9', '#000000'),('#f9cb9c', '#000000', '#ffe5d9', '#000000')]
        for i in range(9): style_matrix.iloc[i, 0] = f'background-color: {colors[i][0]}; color: {colors[i][1]}'; style_matrix.iloc[i, 1] = f'background-color: {colors[i][2]}; color: {colors[i][3]}'
    return style_matrix

def categorize_pieces(nombre_pieces):
    # Version Corrigée (multi-lignes)
    try: nb = int(nombre_pieces)
    except (ValueError, TypeError): return "Autre/Inconnu"
    if nb == 1: return "=1"
    elif nb == 2: return "=2"
    elif nb == 3: return "=3"
    elif nb == 4: return "=4"
    elif nb > 4: return ">4"
    else: return "Autre/Inconnu"

# --- AJOUTER CETTE FONCTION MISE À JOUR ---


def load_clean_moteurimmo(uploaded_file):
    if uploaded_file is None: return pd.DataFrame() # Retourner DF vide si pas de fichier
    logging.info(f"Chargement MoteurImmo unique: {uploaded_file.name}")
    try:
        # Charger le fichier
        df_mi = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', dtype=str)
        logging.info(f"MoteurImmo unique - {len(df_mi)} lignes lues.")

        # --- Nettoyage et Renommage ---
        column_mapping = {
            'Prix/loyer (en €)': 'valeur_fonciere',
            'Surface habitable (en m²)': 'surface_reelle_bati',
            'Type de bien': 'type_local',
            'Nombre de pièces': 'nombre_pieces_principales',
            'Surface du terrain (en m²)': 'surface_terrain',
            'Prix/loyer au m² (en €/m²)': 'prix_m2',
            'Localisation': 'nom_commune',
            'URL': "Lien Annonce"
            # Ajoutez d'autres colonnes sources ici si nécessaire
        }
        required_src_cols = ['Prix/loyer (en €)', 'Surface habitable (en m²)', 'Prix/loyer au m² (en €/m²)', 'Type de bien', 'Localisation'] # Colonnes essentielles
        # S'assurer que la colonne 'Options' est gardée si elle existe
        cols_to_keep_mapped = [col for col in column_mapping.keys() if col in df_mi.columns]
        cols_to_keep_extra = []
        if 'Options' in df_mi.columns:
            cols_to_keep_extra.append('Options')
            logging.info("Colonne 'Options' trouvée et conservée.")

        # Combiner les colonnes mappées et extra
        all_cols_to_keep = list(set(cols_to_keep_mapped + cols_to_keep_extra)) # Utiliser set pour éviter doublons

        # Vérifier les colonnes requises
        missing_src = [col for col in required_src_cols if col not in df_mi.columns]
        if missing_src:
            st.error(f"Colonnes MoteurImmo manquantes: {', '.join(missing_src)}")
            logging.error(f"Colonnes MoteurImmo manquantes: {', '.join(missing_src)} dans {uploaded_file.name}")
            return pd.DataFrame()

        # Sélectionner et renommer
        df_mi = df_mi[all_cols_to_keep].copy()
        df_mi.rename(columns=column_mapping, inplace=True)
        logging.info(f"MoteurImmo - Colonnes après renommage: {list(df_mi.columns)}")

        # Nettoyer commune (inchangé)
        if 'nom_commune' in df_mi.columns:
            df_mi['nom_commune'] = df_mi['nom_commune'].str.replace(r'\s*\(\d+\)\s*$', '', regex=True).str.strip()

        # --- Conversion de Types et Nettoyage Numérique (inchangé) ---
        logging.info("Nettoyage types MoteurImmo...")
        df_mi.replace([np.inf, -np.inf, np.nan], 0, inplace=True) # Remplacer NaN/inf avant conversion
        cols_to_convert_num = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain', 'prix_m2']
        for col in cols_to_convert_num:
            if col in df_mi.columns:
                # Traitement pour supprimer les non-numériques et convertir
                series_str = df_mi[col].astype(str).str.replace(r'[^\d,.]', '', regex=True).str.replace(',', '.', regex=False)
                # Convertir en numérique, mettre NaN en cas d'échec, puis remplir NaN par 0
                df_mi[col] = pd.to_numeric(series_str, errors='coerce').fillna(0)
                # Convertir en Int sauf pour prix_m2
                if col != 'prix_m2':
                    df_mi[col] = df_mi[col].astype(int)
                else:
                     df_mi[col] = df_mi[col].astype(float) # Garder float pour prix/m²


        # Nettoyer et filtrer type_local (inchangé)
        if 'type_local' in df_mi.columns:
            type_map = {'appartement': 'Appartement', 'maison': 'Maison'}
            df_mi['type_local'] = df_mi['type_local'].str.lower().map(type_map).fillna(df_mi['type_local']) # Garde original si non mappé
            df_mi = df_mi[df_mi['type_local'].isin(['Appartement', 'Maison'])]

        # --- NOUVEAU: Classification Neuf/Ancien basée sur 'Options' ---
        if 'Options' in df_mi.columns:
            df_mi['Options'] = df_mi['Options'].astype(str) # Assurer type string
            # Vérifie la présence de 'neuf' (insensible à la casse), gère les NaN éventuels dans Options
            is_neuf = df_mi['Options'].str.contains('neuf', case=False, na=False, regex=False) # regex=False pour performance si simple mot
            df_mi['statut_bien'] = np.where(is_neuf, 'Neuf', 'Ancien')
            logging.info("Classification Neuf/Ancien basée sur colonne 'Options' effectuée.")
            count_neuf = df_mi['statut_bien'].value_counts().get('Neuf', 0)
            logging.info(f"Nombre de biens classifiés 'Neuf': {count_neuf}")
        else:
            # Comportement par défaut si la colonne 'Options' manque
            df_mi['statut_bien'] = 'Ancien' # Hypothèse par défaut
            logging.warning("Colonne 'Options' manquante dans le fichier MoteurImmo. Tous les biens sont considérés comme 'Ancien'.")
        # --- FIN NOUVEAU ---

        logging.info(f"Nettoyage MoteurImmo unique terminé. {len(df_mi)} lignes retournées.")
        return df_mi

    except Exception as e:
        logging.error(f"Erreur load_clean_moteurimmo pour {uploaded_file.name}: {e}", exc_info=True)
        st.error(f"Erreur traitement fichier MoteurImmo '{uploaded_file.name}': {e}")
        return pd.DataFrame()



# Assurez-vous que dataframe_image et matplotlib sont installés et importables
try:
    import dataframe_image as dfi
    DATAFRAME_IMAGE_AVAILABLE = True
    logging.info("Bibliothèque 'dataframe_image' trouvée.")
except ImportError:
    DATAFRAME_IMAGE_AVAILABLE = False
    logging.warning("Bibliothèque 'dataframe_image' NON TROUVÉE. L'export des tableaux en image échouera.")

# --- Helper Function pour créer l'image du tableau stylisé ---
# >>> DÉBUT DU CODE À COPIER/COLLER DANS VOTRE SCRIPT <<<
def create_styled_table_image(df_stats_combined, output_image_path="temp_stats_table.png"):
    """
    Crée une image PNG d'un DataFrame stylisé avec les couleurs de fond.
    Tente d'utiliser le backend 'chrome' puis 'matplotlib' en fallback.
    """
    if not DATAFRAME_IMAGE_AVAILABLE:
        logging.error("Impossible de créer image: 'dataframe_image' non installé.")
        return False

    # Vérifier si matplotlib est disponible (requis pour le fallback au minimum)
    try:
        import matplotlib
    except ImportError:
         logging.error("Impossible de créer image: 'matplotlib' non installé (requis).")
         return False

    if df_stats_combined is None or df_stats_combined.empty:
        logging.warning("DataFrame vide/None fourni pour create_styled_table_image.")
        return False

    logging.info(f"Génération de l'image du tableau stylisé vers {output_image_path}...")

    # --- Logique de Style (identique à celle utilisée dans les onglets Streamlit) ---
    def apply_styling(df): # Reçoit le DataFrame directement
        colors = [
            # Adaptez ces couleurs et cet ordre pour correspondre EXACTEMENT
            # au tableau combiné que vous voulez générer
            ('#000000', '#FFFFFF', '#d9d9d9', '#000000'), # 0: Nb Biens
            ('#6fa8dc', '#FFFFFF', '#cfe2f3', '#000000'), # 1: Moyenne VF
            ('#6fa8dc', '#FFFFFF', '#cfe2f3', '#000000'), # 2: Moyenne PM2
            ('#FFBF00', '#000000', '#FFEBCC', '#000000'), # 3: Médiane VF (Exemple couleur orange)
            ('#FFBF00', '#000000', '#FFEBCC', '#000000'), # 4: Médiane PM2(Exemple couleur orange)
            ('#93c47d', '#FFFFFF', '#d9ead3', '#000000'), # 5: Min VF
            ('#93c47d', '#FFFFFF', '#d9ead3', '#000000'), # 6: Min PM2
            ('#ff0000', '#FFFFFF', '#f4cccc', '#000000'), # 7: Max VF
            ('#ff0000', '#FFFFFF', '#f4cccc', '#000000'), # 8: Max PM2
            ('#f9cb9c', '#000000', '#ffe5d9', '#000000'), # 9: P90 VF
            ('#f9cb9c', '#000000', '#ffe5d9', '#000000')  # 10: P90 PM2
            # Ajoutez/supprimez des lignes de couleur si votre tableau a plus/moins de lignes
        ]
        styler_obj = pd.DataFrame('', index=df.index, columns=df.columns)
        num_rows_to_style = min(len(df), len(colors))
        for i in range(num_rows_to_style):
            try: # Utiliser iloc pour être sûr
                if len(df.columns) > 0: styler_obj.iloc[i, 0] = f'background-color: {colors[i][0]}; color: {colors[i][1]}; text-align: left; font-weight: bold; padding: 3px 5px;'
                if len(df.columns) > 1: styler_obj.iloc[i, 1] = f'background-color: {colors[i][2]}; color: {colors[i][3]}; text-align: right; padding: 3px 5px;'
            except IndexError: # Sécurité si i dépasse les bornes (ne devrait pas arriver avec min())
                logging.warning(f"Index de style {i} hors limites pour le DataFrame.")
                break
        return styler_obj
    # --- Fin Logique de Style ---

    try:
        # Préparer le DataFrame (assurer colonnes Métrique/Valeur)
        if 'Métrique' not in df_stats_combined.columns or 'Valeur' not in df_stats_combined.columns:
            logging.error("DF pour image table doit avoir colonnes 'Métrique' et 'Valeur'.")
            return False # Ne pas essayer de renommer ici, le DF doit être prêt en amont

        df_display = df_stats_combined[['Métrique', 'Valeur']].copy()
        if df_display.empty: logging.warning("DF vide après sélection colonnes."); return False

        # Appliquer le style
        styled_df = df_display.style.hide(axis="index").apply(apply_styling, axis=None)

        # Logique d'export (Try Chrome, fallback Matplotlib)
        try:
            logging.info("Tentative export image table avec backend 'chrome'...")
            dfi.export(styled_df, output_image_path, table_conversion='chrome')
            logging.info("Image table générée avec succès (via Chrome).")
            return True
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as e_chrome:
            logging.warning(f"Échec export table avec 'chrome': {e_chrome}. Tentative avec 'matplotlib'.")
            try:
                import matplotlib # Vérifier à nouveau import
                logging.info("Tentative export image table avec backend 'matplotlib'...")
                dfi.export(styled_df, output_image_path, table_conversion='matplotlib')
                logging.info("Image table générée avec succès (via Matplotlib).")
                return True
            except ImportError: logging.error("'matplotlib' non trouvé."); return False
            except Exception as e_mpl: logging.error(f"Échec export table avec 'matplotlib': {e_mpl}", exc_info=True); return False
        except Exception as e_other: logging.error(f"Erreur inattendue dfi.export: {e_other}", exc_info=True); return False

    except Exception as e_style_prepare:
        logging.error(f"Erreur préparation/style DF pour image: {e_style_prepare}", exc_info=True)
        return False
# >>> FIN DU CODE À COPIER/COLLER DANS VOTRE SCRIPT <<<
# --- Fin Helper Function Table Image ---


# --- Dans report_generator.py ou Analyse_secteur_V6.01.py ---
# (Gardez la fonction create_styled_table_image juste avant)

def export_all_elements_as_zip(elements_to_export: dict):
    """
    Crée une archive zip en mémoire contenant des figures ET des dataframes
    de STATISTIQUES sous forme d'images PNG. Ignore les autres types de DF.
    """
    zip_buffer = io.BytesIO()
    kaleido_available = True
    df_image_available = True
    try: import kaleido
    except ImportError: kaleido_available = False
    try: import dataframe_image as dfi
    except ImportError: df_image_available = False

    temp_table_img_path = "temp_table_for_zip.png"

    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
            logging.info(f"Création ZIP PNG pour {len(elements_to_export)} éléments prévus...")
            added_count = 0
            for filename, element in elements_to_export.items():
                if element is None: continue

                # Assurer .png
                if not filename.lower().endswith(".png"): filename = os.path.splitext(filename)[0] + ".png"

                try:
                    # --- Traitement Figures Plotly ---
                    if hasattr(element, 'to_image'):
                        if not kaleido_available:
                            logging.warning(f"Kaleido manquant, PNG ignoré: {filename}")
                            zip_file.writestr(f"ERREUR_{filename}_KALEIDO_MANQUANT.txt", "Installer Kaleido (pip install -U kaleido)")
                            continue
                        logging.info(f"Ajout Figure -> PNG: {filename}")
                        if hasattr(element, 'update_layout'): element.update_layout(template="plotly_white")
                        img_bytes = element.to_image(format="png", scale=2)
                        zip_file.writestr(filename, img_bytes)
                        added_count += 1

                    # --- Traitement DataFrames (Supposés être Stats pour Image) ---
                    elif isinstance(element, pd.DataFrame):
                        if element.empty: # Ignorer DF vides
                            logging.warning(f"DataFrame '{filename}' vide, ignoré.")
                            continue
                        if not df_image_available:
                            logging.warning(f"dataframe_image manquant, PNG tableau ignoré: {filename}")
                            zip_file.writestr(f"ERREUR_{filename}_DFIMAGE_MANQUANT.txt", "Installer dataframe_image (pip install dataframe_image)")
                            continue

                        logging.info(f"Ajout DataFrame -> PNG: {filename}")
                        # Appeler create_styled_table_image (qui vérifie les colonnes)
                        success_img = create_styled_table_image(element, temp_table_img_path)

                        if success_img and os.path.exists(temp_table_img_path):
                            with open(temp_table_img_path, "rb") as f_img: img_bytes = f_img.read()
                            zip_file.writestr(filename, img_bytes)
                            added_count += 1
                            try: os.remove(temp_table_img_path)
                            except Exception: pass
                        else:
                            logging.error(f"Échec création image pour tableau '{filename}'. Vérifier si colonnes 'Métrique'/'Valeur' présentes.")
                            zip_file.writestr(f"ERREUR_{filename}_IMAGE_TABLEAU.txt", f"Impossible de générer l'image. Vérifier format DF.")

                    else: logging.warning(f"Type non supporté '{filename}', ignoré.")

                except Exception as e_item:
                    logging.error(f"Erreur export élément '{filename}': {e_item}", exc_info=True)
                    zip_file.writestr(f"ERREUR_{filename}.txt", f"Erreur export {filename}:\n{str(e_item)}")

        zip_buffer.seek(0)
        logging.info(f"Buffer Zip créé avec {added_count} éléments.")
        return zip_buffer # Retourne seulement le buffer

    except Exception as e_zip:
        logging.error(f"Erreur création fichier Zip: {e_zip}", exc_info=True)
        return None
    finally:
        if os.path.exists(temp_table_img_path):
            try: os.remove(temp_table_img_path)
            except Exception: pass
# --- Fin fonction Zip (simplifiée) ---



def apply_row_styles(df, colors_list):
    """Applique des styles ligne par ligne en utilisant une liste de couleurs fournie."""
    style_matrix = pd.DataFrame('', index=df.index, columns=df.columns)
    rows_to_style = min(len(df), len(colors_list))
    for i in range(rows_to_style):
        try:
            if len(df.columns) > 0: style_matrix.iloc[i, 0] = f'background-color: {colors_list[i][0]}; color: {colors_list[i][1]}; font-weight: bold; text-align: left; padding: 3px 5px;'
            if len(df.columns) > 1: style_matrix.iloc[i, 1] = f'background-color: {colors_list[i][2]}; color: {colors_list[i][3]}; text-align: right; padding: 3px 5px;'
        except IndexError:
            logging.warning(f"apply_row_styles: Index {i} hors limites.")
            break
    return style_matrix



def apply_common_graph_layout(fig, title_size=24, axis_title_size=20, tick_font_size=16):
    """Applique une mise en page commune (polices) à une figure Plotly."""
    if fig is not None: # Vérifier si la figure existe
        try:
            fig.update_layout(
                title_font_size=title_size,
                xaxis_title_font_size=axis_title_size,
                yaxis_title_font_size=axis_title_size,
                xaxis_tickfont_size=tick_font_size,
                yaxis_tickfont_size=tick_font_size,
                legend_font_size=tick_font_size, # Ajuster si besoin pour la légende
                legend_title_font_size=axis_title_size, # Ajuster si besoin
                font=dict( # Optionnel: Définir une famille de police globale pour le graphe
                    # family="Arial, sans-serif",
                    size=tick_font_size # Taille de police par défaut si non spécifiée ailleurs
                ),
                template="plotly_white" # Assurer un template cohérent
            )
        except Exception as e_layout:
            logging.warning(f"Impossible d'appliquer le layout commun au graphique: {e_layout}")
            # Ne pas planter l'appli si l'update échoue
    return fig # Retourner la figure modifiée (ou l'originale si None/erreur)

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide")

# --- 2. AJOUT POUR MODIFIER LA TAILLE DE LA POLICE (Version avec Variables CSS) ---



# <style> ************************ VELAURS D ORIGINE
# /* --- SECTION PRINCIPALE POUR MODIFIER LES TAILLES --- */
# /* Modifiez les valeurs en 'em' ici (1em = taille normale) */
# :root {
#     --font-size-global: 1.05em;        /* Taille de base pour la plupart des textes */
#     --font-size-titre-h1: 1.8em;      /* Taille pour st.title() */
#     --font-size-titre-h2: 1.5em;      /* Taille pour st.header() */
#     --font-size-titre-h3: 1.3em;      /* Taille pour st.subheader() */
#     --font-size-table-header: 1.1em;  /* Taille pour les en-têtes de colonnes des tables */
#     --font-size-table-cell: 1.1em;    /* Taille pour les valeurs dans les cellules des tables */
#     --font-size-metric-value: 1.5em;  /* Taille pour la valeur principale de st.metric */
#     --font-size-metric-label: 1.0em;  /* Taille pour le label de st.metric */
#     --font-size-button: 1.0em;        /* Taille pour le texte des boutons */
#     --font-size-widget-label: 1.0em;  /* Taille pour les labels des widgets (slider, select...) */
#     --font-size-expander-header: 1.05em;/* Taille pour le titre des st.expander */
#     /* Ajoutez d'autres variables si nécessaire */
# }
# /* --- FIN SECTION PRINCIPALE --- */








if check_password():



    st.markdown("""
    <style>
    /* --- SECTION PRINCIPALE POUR MODIFIER LES TAILLES --- */
    /* Modifiez les valeurs en 'em' ici (1em = taille normale) */
    :root {
        --font-size-global: 1.05em;        /* Taille de base pour la plupart des textes */
        --font-size-titre-h1: 2.8em;      /* Taille pour st.title() */
        --font-size-titre-h2: 1.5em;      /* Taille pour st.header() */
        --font-size-titre-h3: 2.3em;      /* Taille pour st.subheader() */
        --font-size-table-header: 1.1em;  /* Taille pour les en-têtes de colonnes des tables */
        --font-size-table-cell: 1.1em;    /* Taille pour les valeurs dans les cellules des tables */
        --font-size-metric-value: 2.0em;  /* Taille pour la valeur principale de st.metric */
        --font-size-metric-label: 1.0em;  /* Taille pour le label de st.metric */
        --font-size-button: 1.0em;        /* Taille pour le texte des boutons */
        --font-size-widget-label: 1.0em;  /* Taille pour les labels des widgets (slider, select...) */
        --font-size-expander-header: 1.05em;/* Taille pour le titre des st.expander */
        /* Ajoutez d'autres variables si nécessaire */
    }
    /* --- FIN SECTION PRINCIPALE --- */


    /* --- Section des règles CSS (Normalement pas besoin de modifier) --- */
    /* Applique les variables aux éléments correspondants */

    /* Taille globale */
    body, .main {
        font-size: var(--font-size-global) !important;
    }

    /* Titres */
    h1 { font-size: var(--font-size-titre-h1) !important; }
    h2 { font-size: var(--font-size-titre-h2) !important; }
    h3 { font-size: var(--font-size-titre-h3) !important; }
    /* h4, h5, h6 ... si vous les utilisez */

    /* Tableaux (st.dataframe) */
    /* En-têtes de colonnes (sélecteur complexe, peut changer avec versions Streamlit) */
    .stDataFrame [data-testid="stDataFrameResizableHeader"] {
        font-size: var(--font-size-table-header) !important;
    }
    /* Cellules de données */
    .stDataFrame table td div {
        font-size: var(--font-size-table-cell) !important;
    }

    /* Metriques (st.metric) */
    [data-testid="stMetricValue"] {
        font-size: var(--font-size-metric-value) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: var(--font-size-metric-label) !important;
    }

    /* Boutons (st.button) */
    .stButton button {
        font-size: var(--font-size-button) !important;
    }

    /* Labels des widgets (st.slider, st.multiselect, etc.) */
    label.stWidgetLabel {
         font-size: var(--font-size-widget-label) !important;
    }
    /* Texte dans Selectbox/Multiselect */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
         font-size: var(--font-size-widget-label) !important; /* Ou une taille dédiée */
    }


    /* Titre des Expanders (st.expander) */
    .streamlit-expanderHeader p {
        font-size: var(--font-size-expander-header) !important;
    }

    /* Ajoutez d'autres règles ici si vous identifiez d'autres éléments */

    </style>
    """, unsafe_allow_html=True)
    # --- FIN AJOUT CSS ---






    # --- 3. TITRE ET RESTE DU CODE ---
    st.title("Tableau Intéractif de Biens Immobiliers")


    # --- Début: Initialisation de l'état de session ---
    if 'df_base' not in st.session_state: st.session_state['df_base'] = pd.DataFrame()
    if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False
    if 'selected_communes' not in st.session_state: st.session_state['selected_communes'] = []
    if 'selected_sections' not in st.session_state: st.session_state['selected_sections'] = []
    if 'df_ancien' not in st.session_state: st.session_state['df_ancien'] = pd.DataFrame() # Pour MoteurImmo Ancien
    if 'df_neuf' not in st.session_state: st.session_state['df_neuf'] = pd.DataFrame()   # Pour MoteurImmo Neuf
    if 'last_upload_ancien' not in st.session_state: st.session_state['last_upload_ancien'] = None # Pour gérer rechargement
    if 'last_upload_neuf' not in st.session_state: st.session_state['last_upload_neuf'] = None # Pour gérer rechargement
    if 'filtered_df_tab1' not in st.session_state: st.session_state['filtered_df_tab1'] = pd.DataFrame()
    if 'filtered_df_tab3_ancien' not in st.session_state: st.session_state['filtered_df_tab3_ancien'] = pd.DataFrame()
    if 'filtered_df_tab3_neuf' not in st.session_state: st.session_state['filtered_df_tab3_neuf'] = pd.DataFrame()
    if 'filtered_df_tab5' not in st.session_state: st.session_state['filtered_df_tab5'] = pd.DataFrame()
    # ... (peut-être ajouter df_impots et impots_data_loaded ici aussi si ce n'est pas déjà fait)
    if 'df_impots' not in st.session_state: st.session_state['df_impots'] = pd.DataFrame()
    if 'impots_data_loaded' not in st.session_state: st.session_state['impots_data_loaded'] = False

    # --- Initialisation Session State pour Exports ---
    export_keys = [
        # Tab 1
        #'export_df_filtered_t1',
        'export_df_stats_detail_t1', # DataFrames
        'export_fig_vf_yearly_t1', 'export_fig_pm2_yearly_t1', # Barres Annuelles
        'export_fig_hist_vf_t1', 'export_fig_hist_pm2_t1',     # Histogrammes
        'export_fig_scatter_vf_t1', 'export_fig_scatter_pm2_t1', # Scatters
        'export_fig_avg_scatter_vf_t1', 'export_fig_avg_scatter_pm2_t1', # Avg Scatters
        # Tab 2
        'export_stats_apt_t2', 'export_stats_mai_t2',          # Stats (DFs)
        'export_fig_type_bien_t2', 'export_fig_apt_vf_t2', 'export_fig_apt_pm2_t2', # Figs Apt
        'export_fig_house_vf_t2', 'export_fig_house_pm2_t2', # Figs Maison
        'export_fig_apt_scatter_vf_t2', 'export_fig_apt_scatter_pm2_t2', # Scatter Apt
        'export_fig_house_scatter_vf_t2', 'export_fig_house_scatter_pm2_t2',# Scatter Maison
        'export_fig_avg_scatter_vf_apt_t2', 'export_fig_avg_scatter_pm2_apt_t2', # Avg Scatter Apt
        'export_fig_avg_scatter_vf_house_t2', 'export_fig_avg_scatter_pm2_house_t2', # Avg Scatter Maison
        # Tab 3
        'export_df_stats_compare_t3',                          # Stats Compare (DF)
        'export_fig_hist_compare_pm2_t3', 'export_fig_hist_compare_vf_t3', # Histos Compare
        'export_fig_scatter_vf_anc_t3','export_fig_scatter_pm2_anc_t3', # Scatter Ancien
        'export_fig_scatter_vf_neuf_t3','export_fig_scatter_pm2_neuf_t3', # Scatter Neuf
        # Tab 4
        'export_df_competition_summary_t4', 
        #'export_df_competition_detail_t4', # DFs Concu
        'export_fig_apt_t4', 'export_fig_mdv_t4', 'export_fig_mat_t4', # Barres Concu
        # Tab 5
        #'export_df_impots_filtered_t5',
        'export_df_impots_stats_t5', # DFs Impots
        'export_fig_vf_yearly_imp_t5', 'export_fig_pm2_yearly_imp_t5', # Barres An Impots
        'export_fig_hist_vf_imp_t5', 'export_fig_hist_pm2_imp_t5',     # Histos Impots
        'export_fig_scatter_vf_imp_t5', 'export_fig_scatter_pm2_imp_t5', # Scatters Impots
        # Tab 6
        'export_df_stats_synthese_t6',                          # Stats Synthese (DF)
        'export_fig_hist_synthese_t6', 'export_fig_box_synthese_t6', # Graphes Synthese
        'export_fig_bar_stats_t6', 'export_fig_kde_t6'           # Graphes Synthese (suite)
    ]
    for key in export_keys:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame() if 'df' in key else None
    # --- Fin Initialisation Session State ---


    def reset_app_state():
        # (Fonction reset_app_state mise à jour pour inclure df_ancien, df_neuf)
        logging.info("Réinitialisation de l'état...");
        app_state_keys = ['data_loaded', 'df_base', 'selected_communes', 'selected_sections', 'df_ancien', 'df_neuf']
        widget_key_prefixes = ['t1_', 't3_', 'geo_', 'apartments_multiselect_', 'houses_multiselect_'] # Ajout t3_ pour onglet 3
        keys_to_delete = list(app_state_keys)
        for key in list(st.session_state.keys()):
            for prefix in widget_key_prefixes:
                if key.startswith(prefix): keys_to_delete.append(key); break
        for key in set(keys_to_delete):
            if key in st.session_state: del st.session_state[key]; logging.info(f"Clé {key} supprimée.")
        # Réinitialiser les valeurs par défaut
        st.session_state['data_loaded'] = False; st.session_state['df_base'] = pd.DataFrame(); st.session_state['selected_communes'] = []; st.session_state['selected_sections'] = []
        st.session_state['df_ancien'] = pd.DataFrame(); st.session_state['df_neuf'] = pd.DataFrame();
        logging.info("Etat réinitialisé.")
        
    # --- Fin: Initialisation de l'état de session ---

    # --- Sidebar: Chargement et Filtres Géographiques ---
    st.sidebar.header("1. Chargement des Données")
    default_target_directory = r"D:\Users\Seb\Documents\IMMOBILIER\MDB\Outils _ simulations\Export_DVF_Test"
    target_directory_input = st.sidebar.text_input("Répertoire sauvegarde (optionnel):", value=default_target_directory, placeholder="Chemin valide ou vide", help="...")
    target_directory = None
    if target_directory_input:
        if os.path.isdir(target_directory_input):
            target_directory = target_directory_input
            logging.info(f"Répertoire sauvegarde: {target_directory}")
        else:
            st.sidebar.warning(f"Répertoire '{target_directory_input}' invalide.")
    elif not target_directory_input:
        st.sidebar.info("Pas de sauvegarde de fichier.")

    # Uploader DVF
    uploaded_file_dvf = st.sidebar.file_uploader("A. Fichier DVF (Vendus)", type=["csv"], key='upload_dvf', on_change=reset_app_state)

    # NOUVEAU: Uploaders MoteurImmo (Commentés - À activer si besoin pour Tab 3/4)
    st.sidebar.markdown("---")
    #uploaded_file_ancien = st.sidebar.file_uploader("B. Fichier MoteurImmo - ANCIEN (À Vendre)", type=["csv"], key='upload_ancien')
    #uploaded_file_neuf = st.sidebar.file_uploader("C. Fichier MoteurImmo - NEUF (À Vendre)", type=["csv"], key='upload_neuf')


    # --- Logique de chargement des données DVF ---
    # Charger DVF si fichier présent et pas déjà chargé
    if uploaded_file_dvf is not None and not st.session_state.data_loaded:
        with st.spinner("Chargement DVF..."):
            try:
                df_loaded = load_data(uploaded_file_dvf, target_directory=target_directory)
                validate_data(df_loaded, required_columns_app)
                st.session_state['df_base'] = clean_data(df_loaded)
                st.session_state['data_loaded'] = True
                logging.info("Données DVF chargées.")
                # Streamlit gère le rerun après upload et on_change
            except ValueError as ve:
                st.error(f"Erreur validation DVF: {ve}")
                reset_app_state()
            except Exception as e:
                st.error(f"Erreur chargement DVF: {e}")
                logging.error(f"Erreur critique DVF: {e}", exc_info=True)
                reset_app_state()

    # --- Logique principale après chargement DVF ---
    # On ne continue que si les données DVF sont chargées et valides
    if st.session_state.data_loaded and not st.session_state.df_base.empty:
        df_base = st.session_state.df_base

        # --- Filtres Géographiques Sidebar (basés sur DVF) ---
        st.sidebar.header("2. Filtres Géographiques (basés sur DVF)")
        all_communes = sorted(df_base['nom_commune'].unique())
        if not all_communes:
            st.sidebar.warning("Aucune commune trouvée dans les données DVF.")
            selected_communes = [] # Assurer que c'est une liste vide
            selected_sections = []
        else:
            # Widget Communes
            selected_communes_widget = st.sidebar.multiselect(
                "a) Choisissez la ou les communes :",
                all_communes,
                default=st.session_state.get('selected_communes', []),
                key='geo_commune_selector'
            )
            # Gérer changement et rerun si nécessaire
            if set(selected_communes_widget) != set(st.session_state.get('selected_communes', [])):
                st.session_state['selected_communes'] = selected_communes_widget
                st.session_state['selected_sections'] = [] # Reset sections si communes changent
                logging.info(f"Communes MAJ: {selected_communes_widget}. Sections reset.")
                st.rerun()

            selected_communes = st.session_state.get('selected_communes', [])
            selected_sections = [] # Initialiser

    # Widget Sections (si communes sélectionnées)
            if selected_communes:
                # Vérifier que les colonnes nécessaires existent
                if 'section_prefixe' in df_base.columns and 'nom_commune' in df_base.columns:
                    df_temp_communes = df_base[df_base['nom_commune'].isin(selected_communes)].copy()
                    # S'assurer que les colonnes sont de type string pour éviter des erreurs de concaténation
                    df_temp_communes['section_prefixe'] = df_temp_communes['section_prefixe'].astype(str).fillna('?') # Remplacer NaN éventuel par '?'
                    df_temp_communes['nom_commune'] = df_temp_communes['nom_commune'].astype(str).fillna('Inconnue')

                    # --- MODIFIÉ : Préparation des options et mapping ---
                    # Obtenir les paires uniques (section, commune) pour les communes sélectionnées
                    section_commune_pairs = df_temp_communes[['section_prefixe', 'nom_commune']].drop_duplicates().sort_values(by=['nom_commune', 'section_prefixe'])

                    if section_commune_pairs.empty:
                        st.sidebar.warning("Aucune section trouvée pour les communes sélectionnées.")
                        options_display = []
                        section_prefix_map = {} # Dictionnaire pour mapper l'affichage -> préfixe brut
                    else:
                        options_display = []
                        section_prefix_map = {}
                        # Créer les options à afficher
                        if len(selected_communes) > 1:
                            # Format "Section - Commune" si plusieurs communes
                            for _, row in section_commune_pairs.iterrows():
                                display_str = f"{row['section_prefixe']} - {row['nom_commune']}"
                                options_display.append(display_str)
                                section_prefix_map[display_str] = row['section_prefixe']
                        else:
                            # Format "Section" si une seule commune (ou si df contient une seule commune)
                            unique_prefixes = sorted(section_commune_pairs['section_prefixe'].unique())
                            options_display = unique_prefixes
                            for section in options_display:
                                section_prefix_map[section] = section # Le préfixe est la clé et la valeur

                    # --- Gestion de la sélection par défaut ---
                    # Récupérer les préfixes BRUTS stockés dans l'état de session
                    raw_sections_in_state = st.session_state.get('selected_sections', [])
                    # Trouver les chaînes d'affichage correspondantes PARMI LES OPTIONS DISPONIBLES
                    default_display = [disp_str for disp_str, raw_prefix in section_prefix_map.items() if raw_prefix in raw_sections_in_state]

                    # Si la sélection validée est vide (ex: changement de communes), sélectionner tout par défaut
                    if not default_display and options_display:
                        default_display = options_display
                    # --- Fin Gestion Défaut ---

                    # Afficher le multiselect avec les options formatées
                    selected_display_sections = st.sidebar.multiselect(
                        "b) Choisissez la ou les sections cadastrales :",
                        options=options_display, # Utilise la liste formatée
                        default=default_display, # Utilise la liste formatée par défaut
                        key='geo_section_selector_display' # Utilisation d'une clé intermédiaire possible si gestion complexe
                    )

                    # Convertir les sélections affichées (ex: "AA - CommuneX") en préfixes bruts (ex: "AA")
                    selected_sections = sorted(list(set(section_prefix_map[disp_str] for disp_str in selected_display_sections)))

                    # Mettre à jour l'état de session avec les préfixes BRUTS pour le filtrage
                    st.session_state['selected_sections'] = selected_sections
                    # --- FIN MODIFICATIONS ---

                else:
                    st.sidebar.error("Colonnes 'section_prefixe' ou 'nom_commune' manquantes !")
                    selected_sections = []
                    st.session_state['selected_sections'] = []
            else:
                # Reset si aucune commune sélectionnée
                selected_sections = []
                st.session_state['selected_sections'] = []



            # # Widget Sections (si communes sélectionnées)
            # if selected_communes:
            #     if 'section_prefixe' in df_base.columns:
            #         df_temp_communes = df_base[df_base['nom_commune'].isin(selected_communes)]
            #         all_sections = sorted(df_temp_communes['section_prefixe'].unique())
            #         if not all_sections:
            #             st.sidebar.warning("Aucune section trouvée pour les communes sélectionnées.")
            #             selected_sections = []
            #         else:
            #             # Gérer sélection par défaut et validation pour Sections
            #             default_sections = all_sections if not st.session_state.get('selected_sections') else st.session_state['selected_sections']
            #             default_sections = [s for s in default_sections if s in all_sections] # Garder seulement les sections valides
            #             if not default_sections and all_sections: default_sections = all_sections # Sélectionner tout si vide ou invalide

            #             selected_sections = st.sidebar.multiselect(
            #                 "b) Choisissez la ou les sections cadastrales :",
            #                 options=all_sections,
            #                 default=default_sections,
            #                 key='geo_section_selector'
            #             )
            #             # Mettre à jour l'état session sections
            #             st.session_state['selected_sections'] = selected_sections
            #     else:
            #         st.sidebar.error("La colonne 'section_prefixe' est manquante dans le fichier DVF !")
            #         selected_sections = []
            #         st.session_state['selected_sections'] = []
            # else:
            #     st.session_state['selected_sections'] = [] # Assurer reset si aucune commune

        # --- Filtre par Proximité (Optionnel) ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. Filtre par Proximité (Optionnel)")

        target_address = st.sidebar.text_input(
            "Adresse de référence :",
            placeholder="Ex: 1 Rue de la Paix, 75002 Paris",
            key="distance_address"
        )
        # --- MODIFICATION : Remplacer Slider par Number Input pour Mètres ---
        radius_m = st.sidebar.number_input(
            "Rayon de recherche (mètres) :",
            min_value=0,       # Minimum 10 mètres
            max_value=50000,     # Maximum 50 km (ajustez si besoin)
            value=1000,      # Valeur par défaut 1000m (1km)
            step=50,        # Pas de 100 mètres
            key="distance_radius_m", # Nouvelle clé
            help="Distance maximale en MÈTRES autour de l'adresse de référence."
        )
        # --- FIN MODIFICATION ---

        # Utiliser une case à cocher pour ACTIVER le filtre
        activate_distance_filter = st.sidebar.checkbox(
            "Activer le filtre par distance",
            key="distance_activate",
            value=st.session_state.get('distance_filter_active_memory', False)
        )
        st.session_state['distance_filter_active_memory'] = activate_distance_filter

        apply_distance_filter_button = False # Init
        if activate_distance_filter and target_address:
            apply_distance_filter_button = st.sidebar.button("Appliquer le filtre par distance", key="distance_apply")
        elif 'target_coords' in st.session_state and not activate_distance_filter: # Nettoyer si désactivé
            del st.session_state['target_coords']
            if 'target_address_found' in st.session_state: del st.session_state['target_address_found']

        # Initialiser le géocodeur (mis en cache)
        @st.cache_resource
        def get_geolocator():
            return Nominatim(user_agent="streamlit_app_immobilier_seb_v1")

        geolocator = get_geolocator()
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

        # Logique de géocodage (si bouton cliqué)
        if apply_distance_filter_button:
            with st.spinner("Géocodage de l'adresse..."):
                try:
                    location = geocode(target_address, addressdetails=True, country_codes='FR', timeout=10)
                    if location:
                        st.session_state['target_coords'] = (location.latitude, location.longitude)
                        st.session_state['target_address_found'] = location.address
                        st.sidebar.success(f"Adresse trouvée:\n{location.address}\n({location.latitude:.4f}, {location.longitude:.4f})")
                    else:
                        st.sidebar.error("Adresse non trouvée. Vérifiez l'adresse ou essayez une formulation différente.")
                        if 'target_coords' in st.session_state: del st.session_state['target_coords']
                        if 'target_address_found' in st.session_state: del st.session_state['target_address_found']
                except Exception as e:
                    st.sidebar.error(f"Erreur de géocodage : {e}")
                    logging.error(f"Erreur Geocoding: {e}", exc_info=True)
                    if 'target_coords' in st.session_state: del st.session_state['target_coords']
                    if 'target_address_found' in st.session_state: del st.session_state['target_address_found']

        # Afficher l'adresse mémorisée si elle existe et que le filtre est actif
        elif 'target_address_found' in st.session_state and activate_distance_filter:
             st.sidebar.info(f"Adresse mémorisée : {st.session_state['target_address_found']}")


        # --- Application des filtres (Logique Corrigée) ---
        st.sidebar.markdown("---") # Séparateur avant affichage état

        # Commencer avec le DataFrame DVF complet
        df_intermediate = df_base.copy() # Utiliser une variable intermédiaire
        filter_log = [] # Pour suivre les filtres appliqués

        # --- 1. Appliquer filtre Commune/Section (Toujours appliqué si sélections faites) ---
        communes_actives = selected_communes is not None and len(selected_communes) > 0
        sections_actives = selected_sections is not None and len(selected_sections) > 0

        if communes_actives:
            df_intermediate = df_intermediate[df_intermediate['nom_commune'].isin(selected_communes)]
            filter_log.append(f"Communes: {', '.join(selected_communes)}") # Indentation Corrigée
            if sections_actives:
                if 'section_prefixe' in df_intermediate.columns:
                    df_intermediate = df_intermediate[df_intermediate['section_prefixe'].isin(selected_sections)]
                    filter_log.append(f"Sections: {', '.join(selected_sections)}")
                else:
                    st.warning("Impossible de filtrer par section (colonne manquante).")
                    # Conserver les données filtrées par commune seulement
            # Cas où section N'EST PAS active mais commune l'est :
            elif not sections_actives: # 'elif' aligné avec 'if sections_actives:' (Correction indentation/logique)
                st.sidebar.warning("Veuillez sélectionner au moins une section pour les communes choisies.")
                # Optionnel: Décider si on vide df_intermediate ou si on continue avec le filtre commune seul
                # df_intermediate = pd.DataFrame() # Décommenter pour vider

        elif not communes_actives: # Si aucune commune n'est sélectionnée (Correctement aligné)
            st.sidebar.warning("Veuillez sélectionner au moins une commune.")
            df_intermediate = pd.DataFrame() # Vider si pas de commune sélectionnée

        # df_intermediate contient maintenant le résultat du filtrage Commune/Section

        # --- 2. Appliquer le filtre par distance SI activé ET SI on a des données après étape 1 ---
        df_final_geo_filtered = df_intermediate.copy() # Initialiser le DF final avec le résultat précédent

        distance_filter_is_active = st.session_state.get('distance_activate', False)
        target_coords = st.session_state.get('target_coords', None)
     # Récupérer la valeur en mètres depuis le number_input via sa clé
        radius_m_from_input = st.session_state.get('distance_radius_m', 1000) # Utiliser la clé du number_input
        # Convertir en kilomètres pour la comparaison
        radius_km_for_filter = radius_m_from_input / 1000.0




        # Appliquer distance seulement si activé, coords trouvées, ET données intermédiaires non vides
        if distance_filter_is_active and target_coords and not df_intermediate.empty:
            # --- MODIFICATION : Mettre à jour le message d'info ---
            st.sidebar.info(f"Application filtre distance: {radius_km_for_filter:.1f} km ({radius_m_from_input} m) autour de {st.session_state.get('target_address_found', 'adresse trouvée')}")
            # --- FIN MODIFICATION ---
            filter_log.append(f"Distance <= {radius_km_for_filter:.1f} km")

            df_to_filter_distance = df_intermediate.copy() # Travailler sur une copie

            # Vérifier et préparer les colonnes lat/lon
            if 'latitude' not in df_to_filter_distance.columns or 'longitude' not in df_to_filter_distance.columns:
                 st.error("Colonnes 'latitude' ou 'longitude' manquantes pour le filtre distance.")
                 # df_final_geo_filtered reste le résultat des filtres commune/section
            else:
                df_to_filter_distance['latitude'] = pd.to_numeric(df_to_filter_distance['latitude'], errors='coerce')
                df_to_filter_distance['longitude'] = pd.to_numeric(df_to_filter_distance['longitude'], errors='coerce')
                df_to_filter_distance.dropna(subset=['latitude', 'longitude'], inplace=True)

                if not df_to_filter_distance.empty:
                    # Fonction pour calculer la distance (peut être définie ici ou globalement)
                    def calculate_distance_km(row, target_point):
                        prop_point = (row['latitude'], row['longitude'])
                        try:
                            # Utiliser geopy.distance.geodesic importé au début
                            return geopy.distance.geodesic(target_point, prop_point).km
                        except ValueError:
                            return np.nan # Gérer les erreurs de calcul

                    # Calculer les distances
                    with st.spinner(f"Calcul des distances pour {len(df_to_filter_distance)} biens..."):
                        df_to_filter_distance['distance_km'] = df_to_filter_distance.apply(
                            lambda row: calculate_distance_km(row, target_coords),
                            axis=1
                        )

                    # Filtrer par distance et assigner au résultat final
                    df_final_geo_filtered = df_to_filter_distance[df_to_filter_distance['distance_km'] <= radius_km_for_filter].copy()
                    logging.info(f"Filtrage distance appliqué. {len(df_final_geo_filtered)} biens restants.")
                else:
                     st.warning("Aucun bien avec coordonnées valides après filtres commune/section.")
                     df_final_geo_filtered = pd.DataFrame() # Réinitialiser si aucune coordonnée valide

        # --- 3. Afficher l'état des filtres et passer aux onglets ---
        if filter_log:
            st.sidebar.success("Filtres actifs : " + " | ".join(filter_log))
            # Optionnel: Afficher le nombre de résultats après TOUS les filtres
            st.sidebar.write(f"Nombre de biens DVF après filtres: {len(df_final_geo_filtered)}")
        elif not df_base.empty and not communes_actives:
             st.sidebar.warning("Aucun filtre géographique actif (commune manquante).")
        elif not df_base.empty and communes_actives and not sections_actives: # Cas où section non sélectionnée mais commune oui
              st.sidebar.warning("Filtre géographique incomplet (section manquante).")
        # elif not df_base.empty: # Cas où aucune sélection faite (peut être redondant)
        #     st.sidebar.info("Aucun filtre géographique sélectionné.")


        # --- Dans la Sidebar ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Exporter Tout")

        if st.sidebar.button("Télécharger Tous les Éléments (Zip)", key="btn_export_all_zip"):
            if not st.session_state.get('data_loaded', False):
                st.sidebar.warning("Veuillez d'abord charger et filtrer des données.")
            else:
                elements_to_export = {}
                # --- Récupérer tous les éléments depuis st.session_state ---
                logging.info("Collecte des éléments pour l'export Zip...")

                # Faire correspondre les clés session_state aux noms de fichiers souhaités
                key_to_filename_map = {
                    'export_df_filtered_t1': 'Tab1_Donnees_Filtrees.png',
                    'export_df_stats_detail_t1': 'Tab1_Statistiques_Detail.png',
                    'export_fig_vf_yearly_t1' : 'Tab1_Evolution_Annuelle_vf.png',
                    'export_fig_pm2_yearly_t1' : 'Tab1_Evolution_Annuelle_pm2.png',
                    'export_fig_hist_vf_t1': 'Tab1_Histo_VF.png',
                    'export_fig_hist_pm2_t1': 'Tab1_Histo_PM2.png',
                    'export_fig_scatter_vf_t1': 'Tab1_Scatter_VF.png',
                    'export_fig_scatter_pm2_t1': 'Tab1_Scatter_PM2.png',
                    'export_fig_avg_scatter_vf_t1': 'Tab1_AvgScatter_VF.png',
                    'export_fig_avg_scatter_pm2_t1': 'Tab1_AvgScatter_PM2.png',
                    'export_stats_apt_t2': 'Tab2_Stats_Appt.png',
                    'export_stats_mai_t2': 'Tab2_Stats_Maison.png',
                    'export_fig_type_bien_t2': 'Tab2_Bar_TypeBien.png',
                    'export_fig_apt_scatter_vf_t2': 'Tab2_Scatter_VF_Appt.png',
                    'export_fig_apt_scatter_pm2_t2': 'Tab2_Scatter_PM2_Appt.png',
                    'export_fig_house_scatter_vf_t2': 'Tab2_Scatter_VF_Maison.png',
                    'export_fig_house_scatter_pm2_t2': 'Tab2_Scatter_PM2_Maison.png',
                    'export_fig_avg_scatter_vf_apt_t2': 'Tab2_Avg_Scatter_VF_Appt.png',
                    'export_fig_avg_scatter_pm2_apt_t2': 'Tab2_Avg_Scatter_PM2_Appt.png',
                    'export_fig_avg_scatter_vf_house_t2': 'Tab2_Avg_Scatter_PM2_Maison.png',
                    'export_fig_avg_scatter_pm2_house_t2': 'Tab2_Avg_Scatter_PM2_Maison.png',
                    'export_df_stats_compare_t3': 'Tab3_Stats_Comparaison.png',
                    'export_fig_hist_compare_pm2_t3': 'Tab3_Histo_Compare_PM2.png',
                    'export_fig_hist_compare_vf_t3': 'Tab3_Histo_Compare_VF.png',
                    'export_fig_scatter_vf_anc_t3': 'Tab3_Scatter_VF_Ancien.png',
                    'export_fig_scatter_pm2_anc_t3': 'Tab3_Scatter_PM2_Ancien.png',
                    'export_fig_scatter_vf_neuf_t3': 'Tab3_Scatter_VF_Neuf.png',
                    'export_fig_scatter_pm2_neuf_t3': 'Tab3_Scatter_PM2_Neuf.png',
                    'export_df_competition_summary_t4': 'Tab4_Concu_Resume.png',
                    'export_df_competition_detail_t4': 'Tab4_Concu_Detail.png', # Si vous stockez ce détail
                    'export_fig_apt_t4': 'Tab4_Bar_Appt.png',
                    'export_fig_mdv_t4': 'Tab4_Bar_MaisonVillage.png',
                    'export_fig_mat_t4': 'Tab4_Bar_MaisonTerrain.png',
                    'export_df_impots_filtered_t5': 'Tab5_Impots_Filtrees.png',
                    'export_df_impots_stats_t5': 'Tab5_Impots_Stats.png', # Si vous stockez les stats
                    'export_fig_vf_yearly_imp_t5': 'Tab5_Evolution_Annuelle_vf',
                    'export_fig_pm2_yearly_imp_t5': 'Tab5_Evolution_Annuelle_pm2',
                    'export_fig_hist_vf_imp_t5': 'Tab5_Histo_VF.png',
                    'export_fig_hist_pm2_imp_t5': 'Tab5_Histo_PM2.png',
                    'export_fig_scatter_vf_imp_t5': 'Tab5_Scatter_VF.png',
                    'export_fig_scatter_pm2_imp_t5': 'Tab5_Scatter_PM2.png',
                    'export_df_stats_synthese_t6': 'Tab6_Stats_Synthese.png',
                    'export_fig_hist_synthese_t6': 'Tab6_Histo_Synthese.png',
                    'export_fig_box_synthese_t6': 'Tab6_BoxPlot_Synthese.png',
                    'export_fig_bar_stats_t6': 'Tab6_Barres_Stats.png',
                    'export_fig_kde_t6': 'Tab6_KDE_Synthese.png'
                }

                # Construire le dictionnaire uniquement avec les éléments disponibles
                for key, filename in key_to_filename_map.items():
                    element = st.session_state.get(key)
                    # Ajouter seulement si l'élément existe (pas None) et si c'est un DF non vide
                    if element is not None:
                        if not (isinstance(element, pd.DataFrame) and element.empty):
                            elements_to_export[filename] = element
                            logging.info(f" -> Prévu pour Zip (PNG): {filename}")
                        else:
                            logging.warning(f"DataFrame '{key}' vide, ignoré.")


                if not elements_to_export:
                    st.sidebar.warning("Aucun élément à exporter n'a été généré (vérifiez les filtres).")
                else:
                    st.sidebar.info(f"Préparation du fichier zip avec {len(elements_to_export)} éléments...")
                    with st.spinner("Création du fichier Zip en cours..."):
                        # Appeler la fonction (assurez-vous qu'elle est définie)
                        zip_buffer = export_all_elements_as_zip(elements_to_export)

                    if zip_buffer:
                        st.sidebar.download_button(
                            label=f"Télécharger ({len(elements_to_export)} éléments)",
                            data=zip_buffer,
                            file_name="export_analyse_complete.zip",
                            mime="application/zip",
                            key="download_all_zip_final"
                        )
                        st.sidebar.success("Fichier Zip prêt !")
                    else:
                        st.sidebar.error("Erreur lors de la création du fichier Zip.")










        # --- Contenu Principal: Affichage via st.tabs ---
        # On continue seulement si le DataFrame final n'est pas vide ET que les filtres géo de base sont faits
        if df_final_geo_filtered.empty:
            # Adapter le message selon la cause (pourquoi c'est vide)
            if not communes_actives:
                st.info("⬅️ Veuillez sélectionner une commune pour afficher les onglets.")
            elif communes_actives and not sections_actives:
                st.info("⬅️ Veuillez sélectionner au moins une section pour afficher les onglets.")
            else: # Les filtres ont été appliqués mais ont donné 0 résultat
                st.warning("Aucune donnée DVF ne correspond à la combinaison de filtres actuelle (y compris le filtre distance si activé).")
        else:
            # Les données sont prêtes, afficher les onglets
            st.success(f"{len(df_final_geo_filtered)} biens DVF correspondent aux filtres.")

            # --- Définition des Onglets (la version à 6 onglets) ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Tableau de Bord (Vendus)",
                "🌍 Analyse globale (Vendus)",
                "⚖️ Comparaison Marché",
                "🛒 Analyse Concurrence Achat",
                "📄 Données Impôts (PDF)",
                "💡 Synthèse Comparative"
            ])
            # --- Fin Définition Onglets ---



            # # --- Contenu Principal: Affichage via st.tabs ---
            # # On continue seulement si le DataFrame final n'est pas vide ET que les filtres géo de base sont faits
            # if df_final_geo_filtered.empty:
            #     # Adapter le message selon la cause
            #     if not communes_actives or (communes_actives and not sections_actives):
            #          st.info("⬅️ Veuillez sélectionner une commune et au moins une section pour afficher les onglets.")
            #     else: # Les filtres ont été appliqués mais ont donné 0 résultat
            #         st.warning("Aucune donnée DVF ne correspond à la combinaison de filtres actuelle (y compris le filtre distance si activé).")
            # else:
            #     st.success(f"{len(df_final_geo_filtered)} biens DVF correspondent aux filtres.")
            #     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Tableau de Bord (Vendus)", "🌍 Analyse globale (Vendus)", "⚖️ Comparaison Marché", "🛒 Analyse Concurrence Achat", "📄 Données Impôts (PDF)", "💡 Synthèse Comparative"])







            # ===========================================
            # ========= ONGLET 1: Tableau de Bord ========= (Version avec Couleurs Streamlit + Stockage Zip PNG)
            # ===========================================
            with tab1:
                st.header(f"Tableau de Bord ({', '.join(selected_communes)} - Sections: {', '.join(selected_sections)})")
                df_tab1_base = df_final_geo_filtered
                logging.info(f"Tab1: Démarrage avec {len(df_tab1_base)} lignes.")

                # --- Définition des Filtres ---
                col_filter1, col_filter2 = st.columns(2)
                DEFAULT_RANGE = (0, 1); DEFAULT_OPTIONS = []; DEFAULT_RADIO_INDEX = 0

                # --- Colonne Filtres 1 ---
                with col_filter1:
                    st.subheader("Filtres Temporels & Financiers")
                    # Filtre Années (Slider)
                    slider_key_annee = 't1_slider_annee'; annees = []
                    if 'annee' in df_tab1_base.columns: annees = sorted(df_tab1_base['annee'].astype(int).unique())
                    min_slider_an, max_slider_an, calculated_default_annee = DEFAULT_RANGE[0], DEFAULT_RANGE[1]+1, DEFAULT_RANGE
                    if annees:
                        min_an_data = min(annees); max_an_data = max(annees); min_slider_an = min_an_data; max_slider_an = max_an_data; calculated_default_annee = (min_an_data, max_an_data)
                        if min_an_data == max_an_data: max_slider_an = max_an_data + 1
                    else: st.caption("Pas de données annuelles.")

                    # --- Correction pour éviter l'avertissement ---
                    # 1. Obtenir la valeur courante depuis session_state OU utiliser le défaut calculé
                    #    SANS écrire manuellement dans st.session_state ici.
                    current_value_or_default = st.session_state.get(slider_key_annee, calculated_default_annee)

                    # 2. Borner cette valeur par rapport aux limites actuelles des données
                    #    (Cette partie est importante pour s'assurer que la valeur reste valide)
                    bounded_value_annee = (
                        max(min_slider_an, current_value_or_default[0]),
                        min(max_slider_an, current_value_or_default[1])
                    )
                    # Assurer que min <= max après bornage
                    if bounded_value_annee[1] < bounded_value_annee[0]:
                        bounded_value_annee = (bounded_value_annee[0], bounded_value_annee[0])

                    # 3. Créer le slider:
                    #    - 'value' utilise la valeur bornée. Si la clé n'existe pas dans session_state,
                    #      Streamlit l'initialisera avec cette valeur.
                    #    - 'key' lie le widget à st.session_state[slider_key_annee].
                    selected_annees = st.slider(
                        "Années :",
                        min_value=min_slider_an,
                        max_value=max_slider_an,
                        value=bounded_value_annee, # Fournir la valeur initiale / actuelle
                        key=slider_key_annee      # Lier à l'état de session
                    )


                    # if slider_key_annee not in st.session_state: st.session_state[slider_key_annee] = calculated_default_annee
                    # current_value_annee = st.session_state[slider_key_annee]; bounded_value_annee = (max(min_slider_an, current_value_annee[0]), min(max_slider_an, current_value_annee[1]))
                    # if bounded_value_annee[1] < bounded_value_annee[0]: bounded_value_annee = (bounded_value_annee[0], bounded_value_annee[0])
                    # selected_annees = st.slider("Années :", min_value=min_slider_an, max_value=max_slider_an, value=bounded_value_annee, key=slider_key_annee)






                    # Filtre Valeurs Foncières (Number Inputs)
                    filter_key_vf = 't1_filter_vf'; min_key_vf = f"{filter_key_vf}_min"; max_key_vf = f"{filter_key_vf}_max"
                    col_vf = 'valeur_fonciere'; min_vf_data, max_vf_data = 0, 1000000
                    if col_vf in df_tab1_base.columns and not df_tab1_base[col_vf].dropna().empty:
                        vf_valides = df_tab1_base[df_tab1_base[col_vf] > 0][col_vf].dropna()
                        if not vf_valides.empty: min_vf_data = int(vf_valides.min()); max_vf_data = int(vf_valides.max())
                    default_min_vf = st.session_state.get(min_key_vf, min_vf_data); default_max_vf = st.session_state.get(max_key_vf, max_vf_data)
                    default_min_vf = max(min_vf_data, default_min_vf); default_max_vf = min(max_vf_data, default_max_vf)
                    if default_max_vf < default_min_vf: default_max_vf = default_min_vf
                    st.write("Valeurs Foncières (€) :"); vf_col_min, vf_col_max = st.columns(2)
                    with vf_col_min: selected_vf_min = st.number_input("Min", min_value=min_vf_data, max_value=max_vf_data, value=default_min_vf, step=10000, key=min_key_vf, format="%d")
                    with vf_col_max: min_val_for_max_vf = selected_vf_min if selected_vf_min is not None else min_vf_data; selected_vf_max = st.number_input("Max", min_value=min_val_for_max_vf, max_value=max_vf_data, value=max(min_val_for_max_vf, default_max_vf), step=10000, key=max_key_vf, format="%d")

                    # Filtre Prix m² (Number Inputs)
                    filter_key_pm2 = 't1_filter_pm2'; min_key_pm2 = f"{filter_key_pm2}_min"; max_key_pm2 = f"{filter_key_pm2}_max"
                    col_pm2 = 'prix_m2'; min_pm2_data, max_pm2_data = 0, 10000
                    if col_pm2 in df_tab1_base.columns and not df_tab1_base[col_pm2].dropna().empty:
                        pm2_valides = df_tab1_base[df_tab1_base[col_pm2] > 0][col_pm2].dropna()
                        if not pm2_valides.empty: min_pm2_data = int(pm2_valides.min()); max_pm2_data = int(pm2_valides.max())
                    default_min_pm2 = st.session_state.get(min_key_pm2, min_pm2_data); default_max_pm2 = st.session_state.get(max_key_pm2, max_pm2_data)
                    default_min_pm2 = max(min_pm2_data, default_min_pm2); default_max_pm2 = min(max_pm2_data, default_max_pm2)
                    if default_max_pm2 < default_min_pm2: default_max_pm2 = default_min_pm2
                    st.write("Prix m² (€/m²) :"); pm2_col_min, pm2_col_max = st.columns(2)
                    with pm2_col_min: selected_pm2_min = st.number_input("Min", min_value=min_pm2_data, max_value=max_pm2_data, value=default_min_pm2, step=100, key=min_key_pm2, format="%d")
                    with pm2_col_max: min_val_for_max_pm2 = selected_pm2_min if selected_pm2_min is not None else min_pm2_data; selected_pm2_max = st.number_input("Max", min_value=min_val_for_max_pm2, max_value=max_pm2_data, value=max(min_val_for_max_pm2, default_max_pm2), step=100, key=max_key_pm2, format="%d")

                # --- Colonne Filtres 2 ---
                with col_filter2:
                    st.subheader("Filtres Caractéristiques")
                    # Filtre Type de Biens
                    multi_key_type = 't1_multi_type'; types_biens = []
                    if 'type_local' in df_tab1_base.columns: types_biens = sorted(df_tab1_base['type_local'].dropna().unique())
                    
                    # if types_biens:
                    #    calculated_default_type = types_biens
                    #    if multi_key_type not in st.session_state: st.session_state[multi_key_type] = calculated_default_type
                    #    current_selection_type = st.session_state[multi_key_type]; validated_selection_type = [item for item in current_selection_type if item in types_biens]
                    #    if not validated_selection_type and types_biens: validated_selection_type = types_biens
                    #    if validated_selection_type != current_selection_type: st.session_state[multi_key_type] = validated_selection_type # Update state only if validated differs
                    #    selected_types_biens = st.multiselect("Type de Biens :", options=types_biens, default=st.session_state[multi_key_type], key=multi_key_type)
                    # else: selected_types_biens = []; st.caption("Pas de types de biens.")


                    if types_biens:
                       calculated_default_type = types_biens # Default is all available types

                       # --- Correction pour éviter l'avertissement ---
                       # Supprimer l'initialisation manuelle et la validation AVANT la création du widget.
                       # Le widget gérera l'état via sa clé et son argument 'default'.

                       selected_types_biens = st.multiselect(
                           "Type de Biens :",
                           options=types_biens,
                           default=calculated_default_type, # Fournir le défaut pour la 1ère exécution
                           key=multi_key_type # Lier à st.session_state['t1_multi_type']
                       )
                       # --- Fin Correction ---

                       # Note: La logique de validation qui existait avant pourrait être utile APRÈS
                       # la création du widget si les 'types_biens' peuvent changer dynamiquement,
                       # mais elle n'est pas la cause de cet avertissement spécifique.

                    else:
                        selected_types_biens = []
                        # S'assurer que l'état est vidé si les options disparaissent
                        if multi_key_type in st.session_state:
                            st.session_state[multi_key_type] = []
                        st.caption("Pas de types de biens.")



                    # Filtre Présence de Terrain
                    radio_key_terrain = 't1_radio_terrain'; options_terrain = ['Tous', 'Oui', 'Non']; calculated_default_terrain = options_terrain[DEFAULT_RADIO_INDEX]
                    if radio_key_terrain not in st.session_state: st.session_state[radio_key_terrain] = calculated_default_terrain
                    try: current_index_terrain = options_terrain.index(st.session_state[radio_key_terrain])
                    except (ValueError, KeyError): current_index_terrain = DEFAULT_RADIO_INDEX; st.session_state[radio_key_terrain] = options_terrain[current_index_terrain]
                    selected_terrain = st.radio("Présence de Terrain :", options=options_terrain, index=current_index_terrain, key=radio_key_terrain)




                    # Filtre Nombre de Pièces
                    slider_key_pieces = 't1_slider_pieces'; col_pieces = 'nombre_pieces_principales'
                    min_slider_pieces, max_slider_pieces, calculated_default_pieces = DEFAULT_RANGE[0], DEFAULT_RANGE[1]+1, DEFAULT_RANGE
                    if col_pieces in df_tab1_base.columns and not df_tab1_base[col_pieces].empty:
                        pieces_valides = pd.to_numeric(df_tab1_base[col_pieces], errors='coerce').dropna()
                        pieces_valides = pieces_valides[pieces_valides >= 0]
                        if not pieces_valides.empty:
                            min_pieces_data = int(pieces_valides.min()); max_pieces_data = int(pieces_valides.max()); min_slider_pieces = min_pieces_data; max_slider_pieces = max_pieces_data; calculated_default_pieces = (min_pieces_data, max_pieces_data)
                            if min_pieces_data == max_pieces_data: max_slider_pieces = max_pieces_data + 1
                    else: st.caption("Pas de données Nb Pièces.")

                    # if slider_key_pieces not in st.session_state: st.session_state[slider_key_pieces] = calculated_default_pieces
                    # current_value_pieces = st.session_state[slider_key_pieces]; bounded_value_pieces = (max(min_slider_pieces, current_value_pieces[0]), min(max_slider_pieces, current_value_pieces[1]))
                    # if bounded_value_pieces[1] < bounded_value_pieces[0]: bounded_value_pieces = (bounded_value_pieces[0], bounded_value_pieces[0])
                    # selected_pieces = st.slider("Nombre de Pièces :", min_value=min_slider_pieces, max_value=max_slider_pieces, value=bounded_value_pieces, key=slider_key_pieces)

                    # --- Correction pour éviter l'avertissement ---
                    # 1. Lire l'état actuel ou utiliser le défaut calculé
                    current_value_or_default_pieces = st.session_state.get(slider_key_pieces, calculated_default_pieces)

                    # 2. Borner la valeur
                    bounded_value_pieces = (
                        max(min_slider_pieces, current_value_or_default_pieces[0]),
                        min(max_slider_pieces, current_value_or_default_pieces[1])
                    )
                    if bounded_value_pieces[1] < bounded_value_pieces[0]:
                        bounded_value_pieces = (bounded_value_pieces[0], bounded_value_pieces[0])

                    # 3. Créer le slider
                    selected_pieces = st.slider(
                        "Nombre de Pièces :",
                        min_value=min_slider_pieces,
                        max_value=max_slider_pieces,
                        value=bounded_value_pieces, # Utiliser la valeur bornée
                        key=slider_key_pieces      # Lier à l'état
                    )
                    # --- Fin Correction ---

                    # Filtre Surface Bâti
                    filter_key_surf = 't1_filter_surf'; min_key_surf = f"{filter_key_surf}_min"; max_key_surf = f"{filter_key_surf}_max"
                    col_surf = 'surface_reelle_bati'; min_surf_data, max_surf_data = 0, 1000
                    if col_surf in df_tab1_base.columns and not df_tab1_base[col_surf].dropna().empty:
                        surf_valides = df_tab1_base[df_tab1_base[col_surf] > 0][col_surf].dropna()
                        if not surf_valides.empty: min_surf_data = int(surf_valides.min()); max_surf_data = int(surf_valides.max())
                    default_min_surf = st.session_state.get(min_key_surf, min_surf_data); default_max_surf = st.session_state.get(max_key_surf, max_surf_data)
                    default_min_surf = max(min_surf_data, default_min_surf); default_max_surf = min(max_surf_data, default_max_surf)
                    if default_max_surf < default_min_surf: default_max_surf = default_min_surf
                    st.write("Surface Bâti (m²) :"); surf_col_min, surf_col_max = st.columns(2)
                    with surf_col_min: selected_surf_min = st.number_input("Min", min_value=min_surf_data, max_value=max_surf_data, value=default_min_surf, step=5, key=min_key_surf, format="%d")
                    with surf_col_max: min_val_for_max_surf = selected_surf_min if selected_surf_min is not None else min_surf_data; selected_surf_max = st.number_input("Max", min_value=min_val_for_max_surf, max_value=max_surf_data, value=max(min_val_for_max_surf, default_max_surf), step=5, key=max_key_surf, format="%d")

                # --- Appliquer les filtres ---
                filtered_df_tab1 = pd.DataFrame() # Init
                try:
                    # Assurer que les variables de filtre existent (elles sont définies par les widgets ci-dessus)
                    mask_tab1 = pd.Series(True, index=df_tab1_base.index)
                    if 'annee' in df_tab1_base.columns: mask_tab1 &= (df_tab1_base['annee'].between(selected_annees[0], selected_annees[1]))
                    if 'surface_reelle_bati' in df_tab1_base.columns: mask_tab1 &= (df_tab1_base['surface_reelle_bati'].between(selected_surf_min, selected_surf_max))
                    if 'nombre_pieces_principales' in df_tab1_base.columns: mask_tab1 &= (df_tab1_base['nombre_pieces_principales'].between(selected_pieces[0], selected_pieces[1]))
                    if 'valeur_fonciere' in df_tab1_base.columns: mask_tab1 &= (df_tab1_base['valeur_fonciere'].between(selected_vf_min, selected_vf_max))
                    if 'prix_m2' in df_tab1_base.columns: mask_tab1 &= (df_tab1_base['prix_m2'].between(selected_pm2_min, selected_pm2_max))
                    if 'type_local' in df_tab1_base.columns and selected_types_biens: mask_tab1 &= (df_tab1_base['type_local'].isin(selected_types_biens))

                    filtered_df_tab1 = df_tab1_base[mask_tab1].copy()

                    if 'surface_terrain' in filtered_df_tab1.columns:
                        if selected_terrain == 'Oui': filtered_df_tab1 = filtered_df_tab1[filtered_df_tab1['surface_terrain'].fillna(0) > 0]
                        elif selected_terrain == 'Non': filtered_df_tab1 = filtered_df_tab1[filtered_df_tab1['surface_terrain'].fillna(0) == 0]

                    logging.info(f"Tab 1: Filtres appliqués, {len(filtered_df_tab1)} lignes restantes.")
                except Exception as e_filter_tab1:
                    st.error(f"Erreur application filtres Tab 1: {e_filter_tab1}")
                    filtered_df_tab1 = pd.DataFrame(); logging.error(f"Erreur filtrage Tab1: {e_filter_tab1}", exc_info=True)


                # --- AJOUTER LA LIGNE DE STOCKAGE ICI ---
                st.session_state['export_df_filtered_t1'] = filtered_df_tab1.copy()
                logging.info(f"Tab 1: filtered_df_tab1 stocké dans session_state ({len(filtered_df_tab1)} lignes).")
                # --- FIN DE L'AJOUT ---




                # --- Affichages et Graphiques ---
                nombre_biens = len(filtered_df_tab1)

                if nombre_biens > 0:
                    # --- Bloc Calcul Stats + Création DF Combiné + Stockage ---
                    st.subheader("Statistiques Clés")
                    stats_calc_ok = False
                    df_stats_combined_for_export = pd.DataFrame()

                    try:
                        stats_dict = {'Nb Biens': nombre_biens}
                        metrics_list_for_export = []
                        metrics_list_for_export.append({'Métrique': 'Nombre de Biens', 'Valeur': f"{nombre_biens:,}".replace(',', ' ')})

                        vf_series = filtered_df_tab1['valeur_fonciere'].dropna().pipe(lambda s: s[s > 0])
                        pm2_series = filtered_df_tab1['prix_m2'].dropna().pipe(lambda s: s[s > 0])

                        def format_stat(value):
                             if pd.notna(value) and np.isfinite(value):
                                 try: return f"{int(round(value)):,}".replace(',', ' ')
                                 except: return str(value)
                             return 'N/A'

                        stats_dict['Moyenne VF (€)'] = format_stat(vf_series.mean()) if not vf_series.empty else 'N/A'
                        stats_dict['Médiane VF (€)'] = format_stat(vf_series.median()) if not vf_series.empty else 'N/A'
                        stats_dict['Min VF (€)'] = format_stat(vf_series.min()) if not vf_series.empty else 'N/A'
                        stats_dict['Max VF (€)'] = format_stat(vf_series.max()) if not vf_series.empty else 'N/A'
                        try: stats_dict['P90 VF (€)'] = format_stat(vf_series.quantile(0.9)) if not vf_series.empty and len(vf_series) >= 10 else 'N/A'
                        except: stats_dict['P90 VF (€)'] = 'N/A'

                        stats_dict['Moyenne m² (€/m²)'] = format_stat(pm2_series.mean()) if not pm2_series.empty else 'N/A'
                        stats_dict['Médiane m² (€/m²)'] = format_stat(pm2_series.median()) if not pm2_series.empty else 'N/A'
                        stats_dict['Min m² (€/m²)'] = format_stat(pm2_series.min()) if not pm2_series.empty else 'N/A'
                        stats_dict['Max m² (€/m²)'] = format_stat(pm2_series.max()) if not pm2_series.empty else 'N/A'
                        try: stats_dict['P90 m² (€/m²)'] = format_stat(pm2_series.quantile(0.9)) if not pm2_series.empty and len(pm2_series) >= 10 else 'N/A'
                        except: stats_dict['P90 m² (€/m²)'] = 'N/A'

                        export_order = [
                            'Nombre de Biens', 'Moyenne VF (€)', 'Moyenne m² (€/m²)',
                            'Médiane VF (€)', 'Médiane m² (€/m²)', 'Min VF (€)', 'Min m² (€/m²)',
                            'Max VF (€)', 'Max m² (€/m²)', 'P90 VF (€)', 'P90 m² (€/m²)'
                        ]
                        for metric_name in export_order:
                            if metric_name != 'Nombre de Biens':
                                metrics_list_for_export.append({'Métrique': metric_name, 'Valeur': stats_dict.get(metric_name, 'N/A')})

                        df_stats_combined_for_export = pd.DataFrame(metrics_list_for_export)
                        try:
                            df_stats_combined_for_export['Métrique'] = pd.Categorical(df_stats_combined_for_export['Métrique'], categories=export_order, ordered=True)
                            df_stats_combined_for_export = df_stats_combined_for_export.sort_values('Métrique').reset_index(drop=True)
                        except Exception as e_reorder: logging.warning(f"Impossible réordonner DF stats T1: {e_reorder}")
                        df_stats_combined_for_export['Valeur'] = df_stats_combined_for_export['Valeur'].fillna('N/A')
                        stats_calc_ok = True
                        logging.info("Tab 1: df_stats_combined_for_export créé.")

                    except Exception as e_stats_calc:
                        st.error(f"Erreur calcul stats Tab 1: {e_stats_calc}"); stats_calc_ok = False
                        logging.error(f"Erreur stats Tab1: {e_stats_calc}", exc_info=True)

                    # --- Stockage du DataFrame Combiné pour l'export ---
                    if stats_calc_ok:
                        st.session_state['export_df_stats_detail_t1'] = df_stats_combined_for_export.copy()
                        logging.info("DataFrame stats combiné (Tab 1) stocké.")
                    else:
                         st.session_state['export_df_stats_detail_t1'] = None
                    # --- Fin Stockage Stats ---

                    # --- Affichage Streamlit (Utilise stats_dict et recrée DFs pour expander) ---
                    if stats_calc_ok: # Afficher seulement si les stats sont OK
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Nombre de Biens", stats_dict.get('Nb Biens', 'N/A'))
                        mcol2.metric("Moyenne VF (€)", stats_dict.get('Moyenne VF (€)', 'N/A'))
                        mcol3.metric("Moyenne m² (€/m²)", stats_dict.get('Moyenne m² (€/m²)', 'N/A'))

                        with st.expander("Voir les statistiques détaillées (tableaux)"):
                            # Recréer les DFs juste pour affichage
                            stats_detail_1 = {k: stats_dict[k] for k in ['Min VF (€)', 'Min m² (€/m²)'] if k in stats_dict}
                            stats_detail_2 = {k: stats_dict[k] for k in ['Max VF (€)', 'Max m² (€/m²)'] if k in stats_dict}
                            stats_detail_3 = {k: stats_dict[k] for k in ['P90 VF (€)', 'P90 m² (€/m²)', 'Médiane VF (€)', 'Médiane m² (€/m²)'] if k in stats_dict} # Ajouter médianes ici

                            df_disp1 = pd.DataFrame(list(stats_detail_1.items()), columns=['Statistique', 'Valeur'])
                            df_disp2 = pd.DataFrame(list(stats_detail_2.items()), columns=['Statistique', 'Valeur'])
                            df_disp3 = pd.DataFrame(list(stats_detail_3.items()), columns=['Statistique', 'Valeur'])

                            # Appliquer style et afficher
                            try:
                                # Assurez-vous que la fonction apply_row_styles est définie plus haut
                                colors_min = [('#93c47d', '#FFFFFF', '#d9ead3', '#000000'), ('#93c47d', '#FFFFFF', '#d9ead3', '#000000')]
                                colors_max = [('#ff0000', '#FFFFFF', '#f4cccc', '#000000'), ('#ff0000', '#FFFFFF', '#f4cccc', '#000000')]
                                # Adapter l'ordre pour correspondre à stats_detail_3
                                colors_p90_med = [
                                    ('#f9cb9c', '#000000', '#ffe5d9', '#000000'), # P90 VF
                                    ('#f9cb9c', '#000000', '#ffe5d9', '#000000'), # P90 PM2
                                    ('#FFBF00', '#000000', '#FFEBCC', '#000000'), # Médiane VF
                                    ('#FFBF00', '#000000', '#FFEBCC', '#000000')  # Médiane PM2
                                 ]

                                # Vérifier si DFs ne sont pas vides avant de styler/afficher
                                scol1, scol2, scol3 = st.columns(3)
                                with scol1:
                                    if not df_disp1.empty:
                                        styled_df1 = df_disp1.style.apply(apply_row_styles, colors_list=colors_min, axis=None).hide(axis="index")
                                        st.dataframe(styled_df1, use_container_width=True, hide_index=True)
                                    else: st.caption("-")
                                with scol2:
                                    if not df_disp2.empty:
                                        styled_df2 = df_disp2.style.apply(apply_row_styles, colors_list=colors_max, axis=None).hide(axis="index")
                                        st.dataframe(styled_df2, use_container_width=True, hide_index=True)
                                    else: st.caption("-")
                                with scol3:
                                    if not df_disp3.empty:
                                        styled_df3 = df_disp3.style.apply(apply_row_styles, colors_list=colors_p90_med, axis=None).hide(axis="index")
                                        st.dataframe(styled_df3, use_container_width=True, hide_index=True)
                                    else: st.caption("-")

                            except NameError: # Si apply_row_styles n'est pas définie
                                st.error("Fonction 'apply_row_styles' non définie pour l'affichage stylisé.")
                                # Affichage brut fallback
                                scol1, scol2, scol3 = st.columns(3)
                                with scol1: st.dataframe(df_disp1, hide_index=True, use_container_width=True)
                                with scol2: st.dataframe(df_disp2, hide_index=True, use_container_width=True)
                                with scol3: st.dataframe(df_disp3, hide_index=True, use_container_width=True)
                            except Exception as e_style_disp:
                                logging.error(f"Erreur style/affichage tables expander T1: {e_style_disp}")
                                st.warning("Erreur affichage tableaux détaillés stylisés.")
                                # Affichage brut fallback
                                scol1, scol2, scol3 = st.columns(3)
                                with scol1: st.dataframe(df_disp1, hide_index=True, use_container_width=True)
                                with scol2: st.dataframe(df_disp2, hide_index=True, use_container_width=True)
                                with scol3: st.dataframe(df_disp3, hide_index=True, use_container_width=True)

                    # --- Affichage et Stockage des Graphiques ---
                    st.subheader("Évolution Annuelle Moyenne")
                    # ... (Code des graphiques annuels avec stockage session_state) ...
                    try:
                        yearly_stats = filtered_df_tab1.groupby('annee').agg({'valeur_fonciere': 'mean', 'prix_m2': 'mean'}).reset_index()
                        if not yearly_stats.empty:
                            yearly_stats['annee'] = yearly_stats['annee'].astype(str)
                            col_evo1, col_evo2 = st.columns(2)
                            with col_evo1:
                                fig_vf_yearly = px.bar(yearly_stats, x='annee', y='valeur_fonciere', title="Valeur Foncière / An", labels={'valeur_fonciere': 'Valeur Moyenne (€)', 'annee': 'Année'}, color_discrete_sequence=['#1f77b4'], template="plotly_white")
                                # <<< AJOUT POUR LA TAILLE DES POLICES DU GRAPHE >>>
                                fig_vf_yearly.update_layout(
                                title_font_size=22,  # Taille du titre principal
                                xaxis_title_font_size=20, # Taille titre axe X
                                yaxis_title_font_size=20, # Taille titre axe Y
                                font_size=20, # Taille de base pour les graduations (ticks), légende etc.
                                # Vous pouvez aussi cibler spécifiquement :
                                xaxis_tickfont_size=18,
                                yaxis_tickfont_size=18,
                                # legend_font_size=12,
                                # legend_title_font_size=13
                             )
                             # <<< FIN AJOUT >>>
                                st.plotly_chart(fig_vf_yearly, use_container_width=True, key='t1_bar_vf_yearly')
                                st.session_state['export_fig_vf_yearly_t1'] = fig_vf_yearly
                            with col_evo2:
                                fig_pm2_yearly = px.bar(yearly_stats, x='annee', y='prix_m2', title="Prix m² / An", labels={'prix_m2': 'Prix m² Moyen (€/m²)', 'annee': 'Année'}, color_discrete_sequence=['#ff7f0e'], template="plotly_white")
                                 # Appliquer le layout commun APRÈS la création
                                #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                                fig_pm2_yearly = apply_common_graph_layout(fig_pm2_yearly) # Utilise les tailles par défaut de la fonction
                                st.plotly_chart(fig_pm2_yearly, use_container_width=True, key='t1_bar_pm2_yearly')
                                st.session_state['export_fig_pm2_yearly_t1'] = fig_pm2_yearly
                        else: st.warning("Aucune donnée agrégée par an.")
                    except Exception as e: st.error(f"Erreur graphiques annuels: {e}"); logging.error(f"Erreur graphes annuels: {e}", exc_info=True)


                    st.subheader("Distribution des Biens")
                    # ... (Code des histogrammes avec stockage session_state) ...
                    try:
                        col_hist1, col_hist2 = st.columns(2)
                        with col_hist1:
                            fig_hist_vf = px.histogram(filtered_df_tab1, x="valeur_fonciere", color="type_local", title="Par Valeur Foncière", labels={"valeur_fonciere": "Valeur Foncière (€)", "type_local":"Type"}, marginal="box", barmode='group', opacity=0.7, nbins=50, template="plotly_white")
                            fig_hist_vf.update_layout(yaxis_title="Nombre de Biens", bargap=0.1)
                            #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                            fig_hist_vf = apply_common_graph_layout(fig_hist_vf) # Utilise les tailles par défaut de la fonction
                            st.plotly_chart(fig_hist_vf, use_container_width=True, key='t1_hist_vf')
                            st.session_state['export_fig_hist_vf_t1'] = fig_hist_vf
                        with col_hist2:
                            fig_hist_pm2 = px.histogram(filtered_df_tab1, x="prix_m2", color="type_local", title="Par Prix au m²", labels={"prix_m2": "Prix au m² (€/m²)", "type_local":"Type"}, marginal="box", barmode='group', opacity=0.7, nbins=50, template="plotly_white")
                            fig_hist_pm2.update_layout(yaxis_title="Nombre de Biens", bargap=0.1)
                            #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                            fig_hist_pm2 = apply_common_graph_layout(fig_hist_pm2) # Utilise les tailles par défaut de la fonction
                            st.plotly_chart(fig_hist_pm2, use_container_width=True, key='t1_hist_pm2')
                            st.session_state['export_fig_hist_pm2_t1'] = fig_hist_pm2
                    except Exception as e: st.error(f"Erreur histogrammes: {e}"); logging.error(f"Erreur histos T1: {e}", exc_info=True)


                    st.subheader("Surface vs Prix (par Nombre de Pièces)")
                    # ... (Code des nuages de points avec stockage session_state) ...
                    try:
                        filtered_df_tab1_scatter = filtered_df_tab1 # Utiliser le DF déjà filtré
                        if 'nombre_pieces_principales' in filtered_df_tab1_scatter.columns:
                            filtered_df_tab1_scatter['nombre_pieces_cat'] = filtered_df_tab1_scatter['nombre_pieces_principales'].apply(categorize_pieces)
                            piece_order = ["=1", "=2", "=3", "=4", ">4", "Autre/Inconnu"]; valid_piece_cats = [cat for cat in piece_order if cat in filtered_df_tab1_scatter['nombre_pieces_cat'].unique()]
                            col_scatter1, col_scatter2 = st.columns(2)
                            with col_scatter1:
                                fig_scatter_vf = px.scatter(filtered_df_tab1_scatter, x="surface_reelle_bati", y="valeur_fonciere", color="nombre_pieces_cat", title="Surface vs Valeur Foncière", labels={"surface_reelle_bati": "Surface (m²)", "valeur_fonciere": "Valeur (€)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats}, hover_data=['nom_commune','section_prefixe'], template="plotly_white")
                                #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                                fig_scatter_vf = apply_common_graph_layout(fig_scatter_vf) # Utilise les tailles par défaut de la fonction
                                st.plotly_chart(fig_scatter_vf, use_container_width=True, key='t1_scatter_vf')
                                st.session_state['export_fig_scatter_vf_t1'] = fig_scatter_vf
                            with col_scatter2:
                                fig_scatter_pm2 = px.scatter(filtered_df_tab1_scatter, x="surface_reelle_bati", y="prix_m2", color="nombre_pieces_cat", title="Surface vs Prix m²", labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix m² (€/m²)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats}, hover_data=['nom_commune','section_prefixe'], template="plotly_white")
                                #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                                fig_scatter_pm2 = apply_common_graph_layout(fig_scatter_pm2) # Utilise les tailles par défaut de la fonction
                                st.plotly_chart(fig_scatter_pm2, use_container_width=True, key='t1_scatter_pm2')
                                st.session_state['export_fig_scatter_pm2_t1'] = fig_scatter_pm2
                        else: st.warning("Col 'nombre_pieces_principales' manquante pour nuages.")
                    except Exception as e: st.error(f"Erreur nuages de points: {e}"); logging.error(f"Erreur nuages points T1: {e}", exc_info=True)


                    st.subheader("Surface vs Prix MOYENS (par Nombre de Pièces)")
                    st.caption("Ces graphiques montrent la moyenne des prix...")
                    # ... (Code des nuages de points moyens avec stockage session_state) ...
                    df_avg_scatter = pd.DataFrame()
                    avg_scatter_calc_ok = False
                    # S'assurer que filtered_df_tab1_scatter est défini et a la colonne catégorie
                    if 'filtered_df_tab1_scatter' in locals() and 'nombre_pieces_cat' in filtered_df_tab1_scatter.columns:
                        if not filtered_df_tab1_scatter.empty:
                            try:
                                df_avg_scatter = filtered_df_tab1_scatter.groupby(['surface_reelle_bati', 'nombre_pieces_cat'], observed=True).agg(valeur_fonciere_moyenne=('valeur_fonciere', 'mean'), prix_m2_moyen=('prix_m2', 'mean'), count=('valeur_fonciere', 'size')).reset_index()
                                if not df_avg_scatter.empty: avg_scatter_calc_ok = True
                            except Exception as e_avg_calc: st.error(f"Erreur calcul moyennes scatter: {e_avg_calc}"); logging.error(f"Err calc avg scatter T1: {e_avg_calc}", exc_info=True)

                            if avg_scatter_calc_ok:
                                try:
                                    col_avg_scatter1, col_avg_scatter2 = st.columns(2)
                                    # Assurer que valid_piece_cats est défini
                                    if 'valid_piece_cats' not in locals(): valid_piece_cats = sorted(df_avg_scatter['nombre_pieces_cat'].unique())

                                    with col_avg_scatter1:
                                        fig_avg_scatter_vf = px.scatter(df_avg_scatter, x="surface_reelle_bati", y="valeur_fonciere_moyenne", color="nombre_pieces_cat", title="Surface vs VF MOYENNE", labels={"valeur_fonciere_moyenne": "Valeur Moyenne (€)", "surface_reelle_bati": "Surface (m²)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats}, hover_data=['count'], template="plotly_white")
                                        #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                                        fig_avg_scatter_vf = apply_common_graph_layout(fig_avg_scatter_vf) # Utilise les tailles par défaut de la fonction
                                        st.plotly_chart(fig_avg_scatter_vf, use_container_width=True, key='t1_avg_scatter_vf')
                                        st.session_state['export_fig_avg_scatter_vf_t1'] = fig_avg_scatter_vf # STOCKER
                                    with col_avg_scatter2:
                                        fig_avg_scatter_pm2 = px.scatter(df_avg_scatter, x="surface_reelle_bati", y="prix_m2_moyen", color="nombre_pieces_cat", title="Surface vs Prix m² MOYEN", labels={"prix_m2_moyen": "Prix m² Moyen (€/m²)", "surface_reelle_bati": "Surface (m²)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats}, hover_data=['count'], template="plotly_white")
                                        #fig_vf_yearly = apply_common_graph_layout(fig_vf_yearly, title_size=30, axis_title_size=36, tick_font_size=34) # Peut surcharger les défauts
                                        fig_avg_scatter_pm2 = apply_common_graph_layout(fig_avg_scatter_pm2) # Utilise les tailles par défaut de la fonction
                                        st.plotly_chart(fig_avg_scatter_pm2, use_container_width=True, key='t1_avg_scatter_pm2')
                                        st.session_state['export_fig_avg_scatter_pm2_t1'] = fig_avg_scatter_pm2 # STOCKER
                                except Exception as e_avg_plot: st.error(f"Erreur affichage nuages moyens: {e_avg_plot}"); logging.error(f"Err plot avg scatter T1: {e_avg_plot}", exc_info=True)
                            else:
                                st.info("Impossible de calculer/afficher des moyennes pour nuages.")
                                st.session_state['export_fig_avg_scatter_vf_t1'] = None
                                st.session_state['export_fig_avg_scatter_pm2_t1'] = None
                        else: # Si filtered_df_tab1_scatter vide
                            st.info("Aucune donnée pour les nuages de points moyens.")
                            st.session_state['export_fig_avg_scatter_vf_t1'] = None
                            st.session_state['export_fig_avg_scatter_pm2_t1'] = None
                    else: # Si filtered_df_tab1_scatter non défini ou sans colonne catégorie
                        st.info("Prérequis manquants pour nuages moyens.")
                        st.session_state['export_fig_avg_scatter_vf_t1'] = None
                        st.session_state['export_fig_avg_scatter_pm2_t1'] = None
                    # --- Fin MOYENS ---

                else: # Si nombre_biens == 0
                    st.info("Aucune donnée à afficher ou exporter avec les filtres actuels de l'onglet 1.")
                    # Reset session state keys for Tab 1
                    keys_to_reset_t1 = [k for k in export_keys if k.endswith('_t1')]
                    for key in keys_to_reset_t1: st.session_state[key] = None

            # --- Fin de with tab1: ---


            # ===========================================
            # ========= ONGLET 2: Analyse globale ========= (Version avec stockage session_state pour Zip PNG)
            # ===========================================
            with tab2:
                st.header(f"Analyse Globale ({', '.join(selected_communes)} - Sections: {', '.join(selected_sections)})")
                df_tab2_base = df_final_geo_filtered
                apartments_base = df_tab2_base[df_tab2_base['type_local'] == 'Appartement'].copy()
                houses_base = df_tab2_base[df_tab2_base['type_local'] == 'Maison'].copy()
                logging.info(f"Tab2: Démarrage avec {len(apartments_base)} appartements et {len(houses_base)} maisons.")

                # --- Définition Bins/Labels ---
                bins_apartments = [0, 20, 30, 50, 61, 81, float('inf')]; labels_apartments = ['<20 m²', '20-29 m²', '30-49 m²', '50-60 m²', '61-80 m²', '>80 m²']
                bins_houses = [0, 40, 71, 101, float('inf')]; labels_houses = ['<40 m²', '40-70 m²', '71-100 m²', '>100 m²']

                # --- Préparation données validation et filtres interactifs ---
                apartments_validation_data = {}; houses_validation_data = {}
                try:
                    if not apartments_base.empty: apartments_validation_data = create_validation_dropdown(apartments_base.copy(), bins_apartments, labels_apartments)
                    if not houses_base.empty: houses_validation_data = create_validation_dropdown(houses_base.copy(), bins_houses, labels_houses)
                except Exception as e_val: st.error(f"Erreur prépa validation: {e_val}"); logging.error(f"Erreur prépa validation T2: {e_val}", exc_info=True)

                col_filter_apt, col_filter_house = st.columns(2)
                apartments_filtered_list = []; houses_filtered_list = []

                # Filtres interactifs Appartements
                with col_filter_apt:
                    st.subheader("Filtre Prix m² - Appartements");
                    if apartments_validation_data:
                        apartments_base_copy_filter = apartments_base.copy()
                        if not apartments_base_copy_filter.empty:
                            try:
                                # Vérifier si surface_reelle_bati existe avant cut
                                if 'surface_reelle_bati' in apartments_base_copy_filter.columns:
                                    apartments_base_copy_filter['surface_bin'] = pd.cut(pd.to_numeric(apartments_base_copy_filter['surface_reelle_bati'],errors='coerce').fillna(0), bins=bins_apartments, labels=labels_apartments, right=False, include_lowest=True)
                                    for bin_label, prices in apartments_validation_data.items():
                                        if prices and bin_label in apartments_base_copy_filter['surface_bin'].cat.categories:
                                            widget_key = f"apartments_multiselect_{bin_label}"
                                            default_selection = st.session_state.get(widget_key, prices)
                                            default_selection = [p for p in default_selection if p in prices]
                                           # if not default_selection: default_selection = prices # Sélectionner tout si vide --> problème de liste qui se réinitialise quand on la vide
                                            selected_prices_for_bin = st.multiselect(f"Prix m² pour {bin_label}:", prices, default=default_selection, key=widget_key)
                                            if selected_prices_for_bin:
                                                mask_apt = (apartments_base_copy_filter['surface_bin'] == bin_label) & (apartments_base_copy_filter['prix_m2'].isin(selected_prices_for_bin))
                                                apartments_filtered_list.append(apartments_base_copy_filter[mask_apt])
                                else: st.warning("Col 'surface_reelle_bati' manquante pour filtre appartements.")
                            except ValueError as e_cut_apt: st.error(f"Erreur découpage appartements: {e_cut_apt}")
                            except Exception as e_filt_apt: st.error(f"Erreur filtre appartements: {e_filt_apt}")
                        else: st.info("Base appartements vide.")
                    else: st.info("Aucune donnée validation appartements.")

                # Filtres interactifs Maisons
                with col_filter_house:
                    st.subheader("Filtre Prix m² - Maisons")
                    if houses_validation_data:
                        houses_base_copy_filter = houses_base.copy()
                        if not houses_base_copy_filter.empty:
                            try:
                                if 'surface_reelle_bati' in houses_base_copy_filter.columns:
                                    houses_base_copy_filter['surface_bin'] = pd.cut(pd.to_numeric(houses_base_copy_filter['surface_reelle_bati'],errors='coerce').fillna(0), bins=bins_houses, labels=labels_houses, right=False, include_lowest=True)
                                    for bin_label, prices in houses_validation_data.items():
                                        if prices and bin_label in houses_base_copy_filter['surface_bin'].cat.categories:
                                            widget_key = f"houses_multiselect_{bin_label}"
                                            default_selection = st.session_state.get(widget_key, prices)
                                            default_selection = [p for p in default_selection if p in prices]
                                            #if not default_selection: default_selection = prices
                                            selected_prices_for_bin = st.multiselect(f"Prix m² pour {bin_label}:", prices, default=default_selection, key=widget_key)
                                            if selected_prices_for_bin:
                                                mask_house = (houses_base_copy_filter['surface_bin'] == bin_label) & (houses_base_copy_filter['prix_m2'].isin(selected_prices_for_bin))
                                                houses_filtered_list.append(houses_base_copy_filter[mask_house])
                                else: st.warning("Col 'surface_reelle_bati' manquante pour filtre maisons.")
                            except ValueError as e_cut_house: st.error(f"Erreur découpage maisons: {e_cut_house}")
                            except Exception as e_filt_house: st.error(f"Erreur filtre maisons: {e_filt_house}")
                        else: st.info("Base maisons vide.")
                    else: st.info("Aucune donnée validation maisons.")

                # Concaténation des résultats filtrés
                apartments_filtered = pd.concat(apartments_filtered_list).copy() if apartments_filtered_list else pd.DataFrame()
                houses_filtered = pd.concat(houses_filtered_list).copy() if houses_filtered_list else pd.DataFrame()
                df_combined_filtered_tab2 = pd.concat([apartments_filtered, houses_filtered], ignore_index=True)
                logging.info(f"Tab 2: Après filtres interactifs, {len(df_combined_filtered_tab2)} lignes combinées.")

                # --- Affichages et Graphiques ---
                nombre_biens_tab2 = len(df_combined_filtered_tab2)
                st.metric("Nombre total de biens trouvés (après filtres prix/m²)", f"{nombre_biens_tab2:,}".replace(',', ' '))
                st.markdown("---")

                if nombre_biens_tab2 > 0:
                    # --- Métriques Globales ---
                    st.subheader("Statistiques Globales (Basées sur Filtres Prix m²)")
                    col_metric_apt, col_metric_house = st.columns(2)
                    apt_mean_pm2 = np.nan # Initialiser pour le calcul d'écart
                    house_mean_pm2 = np.nan # Initialiser pour le calcul d'écart

                    with col_metric_apt:
                        st.markdown("<h5 style='text-align: center;'>Appartements</h5>", unsafe_allow_html=True)
                        if not apartments_filtered.empty:
                            try:
                                nb_apt = len(apartments_filtered)
                                st.metric("Nombre d'Appartements", f"{nb_apt:,}".replace(',', ' ')) # Ajout de la métrique ici
                                
                                vf_apt = apartments_filtered['valeur_fonciere'].dropna().pipe(lambda s: s[s > 0])
                                pm2_apt = apartments_filtered['prix_m2'].dropna().pipe(lambda s: s[s > 0])
                                apt_mean_vf = vf_apt.mean() if not vf_apt.empty else np.nan
                                apt_median_vf = vf_apt.median() if not vf_apt.empty else np.nan
                                apt_mean_pm2 = pm2_apt.mean() if not pm2_apt.empty else np.nan # Stocker valeur numérique
                                apt_median_pm2 = pm2_apt.median() if not pm2_apt.empty else np.nan

                                sub_col_apt1, sub_col_apt2 = st.columns(2)
                                with sub_col_apt1: st.metric("VF Moyenne", f"{apt_mean_vf:,.0f} €".replace(',', ' ') if pd.notna(apt_mean_vf) else "N/A")
                                with sub_col_apt2: st.metric("Prix/m² Moyen", f"{apt_mean_pm2:,.0f} €/m²".replace(',', ' ') if pd.notna(apt_mean_pm2) else "N/A")
                                sub_col_apt3, sub_col_apt4 = st.columns(2)
                                with sub_col_apt3: st.metric("VF Médiane", f"{apt_median_vf:,.0f} €".replace(',', ' ') if pd.notna(apt_median_vf) else "N/A")
                                with sub_col_apt4: st.metric("Prix/m² Médian", f"{apt_median_pm2:,.0f} €/m²".replace(',', ' ') if pd.notna(apt_median_pm2) else "N/A")
                            except Exception as e_met_apt: st.error(f"Err calcul métriques Apt: {e_met_apt}"); logging.error(f"Err métriques Apt T2: {e_met_apt}", exc_info=True)
                        else: 
                            st.metric("Nombre d'Appartements", 0)
                            st.info("Aucun appartement.")

                    with col_metric_house:
                        st.markdown("<h5 style='text-align: center;'>Maisons</h5>", unsafe_allow_html=True)
                        if not houses_filtered.empty:
                            try:
                                nb_mai = len(houses_filtered)
                                st.metric("Nombre de Maisons", f"{nb_mai:,}".replace(',', ' ')) # Ajout de la métrique ici

                                vf_house = houses_filtered['valeur_fonciere'].dropna().pipe(lambda s: s[s > 0])
                                pm2_house = houses_filtered['prix_m2'].dropna().pipe(lambda s: s[s > 0])
                                house_mean_vf = vf_house.mean() if not vf_house.empty else np.nan
                                house_median_vf = vf_house.median() if not vf_house.empty else np.nan
                                house_mean_pm2 = pm2_house.mean() if not pm2_house.empty else np.nan # Stocker valeur numérique
                                house_median_pm2 = pm2_house.median() if not pm2_house.empty else np.nan

                                sub_col_house1, sub_col_house2 = st.columns(2)
                                with sub_col_house1: st.metric("VF Moyenne", f"{house_mean_vf:,.0f} €".replace(',', ' ') if pd.notna(house_mean_vf) else "N/A")
                                with sub_col_house2: st.metric("Prix/m² Moyen", f"{house_mean_pm2:,.0f} €/m²".replace(',', ' ') if pd.notna(house_mean_pm2) else "N/A")
                                sub_col_house3, sub_col_house4 = st.columns(2)
                                with sub_col_house3: st.metric("VF Médiane", f"{house_median_vf:,.0f} €".replace(',', ' ') if pd.notna(house_median_vf) else "N/A")
                                with sub_col_house4: st.metric("Prix/m² Médian", f"{house_median_pm2:,.0f} €/m²".replace(',', ' ') if pd.notna(house_median_pm2) else "N/A")
                            except Exception as e_met_house: st.error(f"Err calcul métriques Maisons: {e_met_house}"); logging.error(f"Err métriques Maisons T2: {e_met_house}", exc_info=True)
                        else: 
                            st.metric("Nombre de Maisons", 0)
                            st.info("Aucune maison.")
                    st.markdown("---")

                    # --- Tableaux Stats Validées ---
                    st.subheader("Statistiques par Tranche (Basées sur Filtres Prix m²)")
                    col_stats_apt, col_stats_house = st.columns(2)
                    with col_stats_apt:
                        st.write("**Appartements**")
                        stats_validated_apartments = pd.DataFrame() # Init
                        try:
                            stats_validated_apartments = calculate_validated_stats(apartments_validation_data, st.session_state, key_prefix="apartments")
                            if pd.notna(apt_mean_pm2) and not stats_validated_apartments.empty and 'Prix_moyen_m2' in stats_validated_apartments.columns:
                                stats_validated_apartments['Prix_moyen_m2'] = pd.to_numeric(stats_validated_apartments['Prix_moyen_m2'], errors='coerce')
                                stats_validated_apartments['Écart / Moyenne Globale (€/m²)'] = stats_validated_apartments['Prix_moyen_m2'] - apt_mean_pm2
                                stats_validated_apartments['Écart / Moyenne Globale (€/m²)'] = stats_validated_apartments['Écart / Moyenne Globale (€/m²)'].apply(lambda x: f"{x:+,.0f}".replace(',', ' ') if pd.notna(x) else "N/A")
                            elif not stats_validated_apartments.empty: stats_validated_apartments['Écart / Moyenne Globale (€/m²)'] = "N/A"
                            st.dataframe(stats_validated_apartments, hide_index=True, use_container_width=True)
                            # STOCKAGE pour Export Zip
                            st.session_state['export_stats_apt_t2'] = stats_validated_apartments.copy()
                            logging.info("Tab 2: stats_validated_apartments stocké.")
                        except Exception as e_stats_apt: st.error(f"Err tableau stats apt: {e_stats_apt}"); logging.error(f"Err table stats apt T2: {e_stats_apt}", exc_info=True); st.session_state['export_stats_apt_t2'] = None

                    with col_stats_house:
                        st.write("**Maisons**")
                        stats_validated_houses = pd.DataFrame() # Init
                        try:
                            stats_validated_houses = calculate_validated_stats(houses_validation_data, st.session_state, key_prefix="houses")
                            if pd.notna(house_mean_pm2) and not stats_validated_houses.empty and 'Prix_moyen_m2' in stats_validated_houses.columns:
                                stats_validated_houses['Prix_moyen_m2'] = pd.to_numeric(stats_validated_houses['Prix_moyen_m2'], errors='coerce')
                                stats_validated_houses['Écart / Moyenne Globale (€/m²)'] = stats_validated_houses['Prix_moyen_m2'] - house_mean_pm2
                                stats_validated_houses['Écart / Moyenne Globale (€/m²)'] = stats_validated_houses['Écart / Moyenne Globale (€/m²)'].apply(lambda x: f"{x:+,.0f}".replace(',', ' ') if pd.notna(x) else "N/A")
                            elif not stats_validated_houses.empty: stats_validated_houses['Écart / Moyenne Globale (€/m²)'] = "N/A"
                            st.dataframe(stats_validated_houses, hide_index=True, use_container_width=True)
                            # STOCKAGE pour Export Zip
                            st.session_state['export_stats_mai_t2'] = stats_validated_houses.copy()
                            logging.info("Tab 2: stats_validated_houses stocké.")
                        except Exception as e_stats_house: st.error(f"Err tableau stats maisons: {e_stats_house}"); logging.error(f"Err table stats maisons T2: {e_stats_house}", exc_info=True); st.session_state['export_stats_mai_t2'] = None

                    # --- Graphe Répartition Types Biens ---
                    st.subheader("Répartition Types Biens (Filtrés Prix m²)")
                    fig_type_bien_filtered = None # Init
                    if not df_combined_filtered_tab2.empty:
                        try:
                            # Vérifier si les colonnes nécessaires existent
                            if {'type_local', 'nombre_pieces_principales', 'annee'}.issubset(df_combined_filtered_tab2.columns):

                                # --- DÉFINITION LOCALE de la fonction (comme dans votre code qui fonctionnait) ---
                                def create_type_bien_safely(row):
                                    try:
                                        # Tenter de convertir en entier, gérer NaN/erreurs
                                        pieces = pd.to_numeric(row['nombre_pieces_principales'], errors='coerce')
                                        if pd.isna(pieces): raise ValueError("Non numérique")
                                        pieces = int(pieces)
                                        # Retourner le type concaténé avec le nombre de pièces
                                        return f"{row.get('type_local', 'Inconnu')}{pieces}" # Utiliser .get pour type_local aussi
                                    except:
                                        # Fallback si conversion échoue ou si type_local manque
                                        type_loc = row.get('type_local', 'Inconnu')
                                        return f"{type_loc}_pièces_inconnues"
                                # --- FIN DÉFINITION LOCALE ---

                                # Appliquer CETTE fonction locale pour créer la colonne
                                df_combined_filtered_tab2['type_bien'] = df_combined_filtered_tab2.apply(create_type_bien_safely, axis=1)

                                if 'type_bien' in df_combined_filtered_tab2.columns:
                                    type_bien_counts_filtered = df_combined_filtered_tab2.groupby(['annee', 'type_bien']).size().reset_index(name='counts')
                                    if not type_bien_counts_filtered.empty:
                                        type_bien_counts_filtered['annee'] = type_bien_counts_filtered['annee'].astype(str)
                                        # Ajout template
                                        fig_type_bien_filtered = px.bar(
                                            type_bien_counts_filtered,
                                            x='annee', y='counts', color='type_bien',
                                            title="Répartition Annuelle (Biens Filtrés par Prix)",
                                            labels={'counts': 'Nombre', 'annee': 'Année', 'type_bien': 'Type et Nb Pièces'}, # Label adapté
                                            barmode='group',
                                            template="plotly_white" # Garder le thème
                                        )
                                        st.plotly_chart(fig_type_bien_filtered, use_container_width=True, key='t2_bar_type_bien_filtered')
                                    else: st.warning("Aucune donnée agrégée pour Type Bien.")
                                else: st.error("La colonne 'type_bien' n'a pas pu être créée.")
                            else: st.warning("Colonnes manquantes ('type_local', 'nombre_pieces_principales', 'annee') pour graphe Type Bien.")
                        except Exception as e_tb: st.error(f"Erreur graphe type bien: {e_tb}"); logging.error(f"Erreur graphe type bien T2: {e_tb}", exc_info=True)
                    else: st.info("Aucun bien combiné (pour graphe Type Bien).")
                    # STOCKAGE pour Export Zip (inchangé)
                    st.session_state['export_fig_type_bien_t2'] = fig_type_bien_filtered
                    # --- Fin Graphe Répartition ---

                    # --- Graphes Évolution Appartements ---
                    st.subheader("Évolution Annuelle - Appartements (Filtrés Prix m²)")
                    fig_apt_vf = None; fig_apt_pm2 = None # Init
                    if not apartments_filtered.empty:
                        try:
                            apartment_stats = apartments_filtered.groupby('annee').agg({'valeur_fonciere': 'mean', 'prix_m2': 'mean'}).reset_index()
                            if not apartment_stats.empty:
                                apartment_stats['annee'] = apartment_stats['annee'].astype(str)
                                col_apt1, col_apt2 = st.columns(2)
                                with col_apt1:
                                    fig_apt_vf = px.bar(apartment_stats, x='annee', y='valeur_fonciere', title="VF Moyenne / An", labels={'valeur_fonciere': 'Valeur (€)', 'annee': 'Année'}, color_discrete_sequence=['#1f77b4'], template="plotly_white")
                                    st.plotly_chart(fig_apt_vf, use_container_width=True, key='t2_bar_apt_vf')
                                with col_apt2:
                                    fig_apt_pm2 = px.bar(apartment_stats, x='annee', y='prix_m2', title="Prix m² Moyen / An", labels={'prix_m2': 'Prix m² (€/m²)', 'annee': 'Année'}, color_discrete_sequence=['#ff7f0e'], template="plotly_white")
                                    st.plotly_chart(fig_apt_pm2, use_container_width=True, key='t2_bar_apt_pm2')
                            else: st.warning("Aucune stat annuelle pour appartements filtrés.")
                        except Exception as e_apt_evo: st.error(f"Err graphes évo apt: {e_apt_evo}"); logging.error(f"Err graphes évo apt T2: {e_apt_evo}", exc_info=True)
                    else: st.info("Aucun appartement après filtre prix (pour graphes évolution).")
                    # STOCKAGE pour Export Zip
                    st.session_state['export_fig_apt_vf_t2'] = fig_apt_vf
                    st.session_state['export_fig_apt_pm2_t2'] = fig_apt_pm2

                    # --- Graphes Évolution Maisons ---
                    st.subheader("Évolution Annuelle - Maisons (Filtrées Prix m²)")
                    fig_house_vf = None; fig_house_pm2 = None # Init
                    if not houses_filtered.empty:
                        try:
                            house_stats = houses_filtered.groupby('annee').agg({'valeur_fonciere': 'mean', 'prix_m2': 'mean'}).reset_index()
                            if not house_stats.empty:
                                house_stats['annee'] = house_stats['annee'].astype(str)
                                col_house1, col_house2 = st.columns(2)
                                with col_house1:
                                    fig_house_vf = px.bar(house_stats, x='annee', y='valeur_fonciere', title="VF Moyenne / An", labels={'valeur_fonciere': 'Valeur (€)', 'annee': 'Année'}, color_discrete_sequence=['#2ca02c'], template="plotly_white")
                                    st.plotly_chart(fig_house_vf, use_container_width=True, key='t2_bar_house_vf')
                                with col_house2:
                                    fig_house_pm2 = px.bar(house_stats, x='annee', y='prix_m2', title="Prix m² Moyen / An", labels={'prix_m2': 'Prix m² (€/m²)', 'annee': 'Année'}, color_discrete_sequence=['#d62728'], template="plotly_white")
                                    st.plotly_chart(fig_house_pm2, use_container_width=True, key='t2_bar_house_pm2')
                            else: st.warning("Aucune stat annuelle pour maisons filtrées.")
                        except Exception as e_house_evo: st.error(f"Err graphes évo maisons: {e_house_evo}"); logging.error(f"Err graphes évo maisons T2: {e_house_evo}", exc_info=True)
                    else: st.info("Aucune maison après filtre prix (pour graphes évolution).")
                    # STOCKAGE pour Export Zip
                    st.session_state['export_fig_house_vf_t2'] = fig_house_vf
                    st.session_state['export_fig_house_pm2_t2'] = fig_house_pm2

                    # --- Nuages de Points Appartements ---
                    st.subheader("Nuages de Points - Appartements (Filtrés Prix m²)")
                    fig_apt_scatter_vf = None; fig_apt_scatter_pm2 = None # Init
                    if not apartments_filtered.empty:
                        try:
                            apartments_filtered_scatter = apartments_filtered.copy()
                            if 'nombre_pieces_principales' in apartments_filtered_scatter.columns:
                                apartments_filtered_scatter['pieces_cat'] = apartments_filtered_scatter['nombre_pieces_principales'].apply(categorize_pieces)
                                piece_order_apt = ["=1", "=2", "=3", "=4", ">4", "Autre/Inconnu"]; valid_piece_cats_apt = [cat for cat in piece_order_apt if cat in apartments_filtered_scatter['pieces_cat'].unique()]
                                col_scatter_apt1, col_scatter_apt2 = st.columns(2)
                                with col_scatter_apt1:
                                    fig_apt_scatter_vf = px.scatter(apartments_filtered_scatter, x="surface_reelle_bati", y="valeur_fonciere", color="pieces_cat", title="Surface vs Valeur", labels={"surface_reelle_bati": "Surface (m²)", "valeur_fonciere": "Valeur (€)", "pieces_cat": "Nb Pièces"}, category_orders={"pieces_cat": valid_piece_cats_apt}, hover_data=['nom_commune','section_prefixe'], template="plotly_white")
                                    st.plotly_chart(fig_apt_scatter_vf, use_container_width=True, key='t2_scatter_apt_vf')
                                with col_scatter_apt2:
                                    fig_apt_scatter_pm2 = px.scatter(apartments_filtered_scatter, x="surface_reelle_bati", y="prix_m2", color="pieces_cat", title="Surface vs Prix m²", labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix m² (€/m²)", "pieces_cat": "Nb Pièces"}, category_orders={"pieces_cat": valid_piece_cats_apt}, hover_data=['nom_commune','section_prefixe'], template="plotly_white")
                                    st.plotly_chart(fig_apt_scatter_pm2, use_container_width=True, key='t2_scatter_apt_pm2')
                            else: st.warning("Colonne 'nombre_pieces_principales' manquante.")
                        except Exception as e_apt_scatter: st.error(f"Err nuages points apt: {e_apt_scatter}"); logging.error(f"Err nuages points apt T2: {e_apt_scatter}", exc_info=True)
                    else: st.info("Aucun appartement après filtre prix (pour nuages).")
                    # STOCKAGE pour Export Zip
                    st.session_state['export_fig_apt_scatter_vf_t2'] = fig_apt_scatter_vf
                    st.session_state['export_fig_apt_scatter_pm2_t2'] = fig_apt_scatter_pm2

                    # --- Nuages de Points Maisons ---
                    st.subheader("Nuages de Points - Maisons (Filtrées Prix m²)")
                    fig_house_scatter_vf = None; fig_house_scatter_pm2 = None # Init
                    if not houses_filtered.empty:
                        try:
                            houses_filtered_scatter = houses_filtered.copy()
                            if 'nombre_pieces_principales' in houses_filtered_scatter.columns:
                                houses_filtered_scatter['pieces_cat'] = houses_filtered_scatter['nombre_pieces_principales'].apply(categorize_pieces)
                                piece_order_house = ["=1", "=2", "=3", "=4", ">4", "Autre/Inconnu"]; valid_piece_cats_house = [cat for cat in piece_order_house if cat in houses_filtered_scatter['pieces_cat'].unique()]
                                col_scatter_house1, col_scatter_house2 = st.columns(2)
                                with col_scatter_house1:
                                    fig_house_scatter_vf = px.scatter(houses_filtered_scatter, x="surface_reelle_bati", y="valeur_fonciere", color="pieces_cat", title="Surface vs Valeur", labels={"surface_reelle_bati": "Surface (m²)", "valeur_fonciere": "Valeur (€)", "pieces_cat": "Nb Pièces"}, category_orders={"pieces_cat": valid_piece_cats_house}, hover_data=['nom_commune','section_prefixe'], template="plotly_white")
                                    st.plotly_chart(fig_house_scatter_vf, use_container_width=True, key='t2_scatter_house_vf')
                                with col_scatter_house2:
                                    fig_house_scatter_pm2 = px.scatter(houses_filtered_scatter, x="surface_reelle_bati", y="prix_m2", color="pieces_cat", title="Surface vs Prix m²", labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix m² (€/m²)", "pieces_cat": "Nb Pièces"}, category_orders={"pieces_cat": valid_piece_cats_house}, hover_data=['nom_commune','section_prefixe'], template="plotly_white")
                                    st.plotly_chart(fig_house_scatter_pm2, use_container_width=True, key='t2_scatter_house_pm2')
                            else: st.warning("Colonne 'nombre_pieces_principales' manquante.")
                        except Exception as e_house_scatter: st.error(f"Err nuages points maisons: {e_house_scatter}"); logging.error(f"Err nuages points maisons T2: {e_house_scatter}", exc_info=True)
                    else: st.info("Aucune maison après filtre prix (pour nuages).")
                    # STOCKAGE pour Export Zip
                    st.session_state['export_fig_house_scatter_vf_t2'] = fig_house_scatter_vf
                    st.session_state['export_fig_house_scatter_pm2_t2'] = fig_house_scatter_pm2


                    # --- AJOUT : Nuages de Points MOYENS - Appartements ---
                    st.subheader("Surface vs Prix MOYENS - Appartements")
                    st.caption("Ces graphiques montrent la moyenne des prix pour les appartements ayant la même surface et catégorie de pièces.")
                    df_avg_scatter_apt = pd.DataFrame()
                    avg_scatter_apt_calc_ok = False
                    # Utiliser apartments_filtered_scatter qui contient déjà 'pieces_cat'
                    if 'apartments_filtered_scatter' in locals() and not apartments_filtered_scatter.empty:
                        try:
                            df_avg_scatter_apt = apartments_filtered_scatter.groupby(['surface_reelle_bati', 'pieces_cat'], observed=True).agg(
                                valeur_fonciere_moyenne=('valeur_fonciere', 'mean'),
                                prix_m2_moyen=('prix_m2', 'mean'),
                                count=('valeur_fonciere', 'size') # Compter le nb de biens par groupe
                            ).reset_index()
                            if not df_avg_scatter_apt.empty:
                                avg_scatter_apt_calc_ok = True
                        except Exception as e_avg_calc_apt:
                            st.error(f"Erreur calcul moyennes scatter (Appt): {e_avg_calc_apt}")
                            logging.error(f"Err calc avg scatter Apt T2: {e_avg_calc_apt}", exc_info=True)

                    fig_avg_scatter_vf_apt = None
                    fig_avg_scatter_pm2_apt = None
                    if avg_scatter_apt_calc_ok:
                        try:
                            col_avg_scatter_apt1, col_avg_scatter_apt2 = st.columns(2)
                            # S'assurer que valid_piece_cats_apt est défini (il l'est par le bloc scatter précédent)
                            if 'valid_piece_cats_apt' not in locals(): valid_piece_cats_apt = sorted(df_avg_scatter_apt['pieces_cat'].unique())

                            with col_avg_scatter_apt1:
                                fig_avg_scatter_vf_apt = px.scatter(
                                    df_avg_scatter_apt, x="surface_reelle_bati", y="valeur_fonciere_moyenne",
                                    color="pieces_cat", title="Surface vs VF MOYENNE",
                                    labels={"valeur_fonciere_moyenne": "Valeur Moyenne (€)", "surface_reelle_bati": "Surface (m²)", "pieces_cat": "Nb Pièces"},
                                    category_orders={"pieces_cat": valid_piece_cats_apt}, hover_data=['count'], template="plotly_white"
                                )
                                st.plotly_chart(fig_avg_scatter_vf_apt, use_container_width=True, key='t2_avg_scatter_apt_vf')
                            with col_avg_scatter_apt2:
                                fig_avg_scatter_pm2_apt = px.scatter(
                                    df_avg_scatter_apt, x="surface_reelle_bati", y="prix_m2_moyen",
                                    color="pieces_cat", title="Surface vs Prix m² MOYEN",
                                    labels={"prix_m2_moyen": "Prix m² Moyen (€/m²)", "surface_reelle_bati": "Surface (m²)", "pieces_cat": "Nb Pièces"},
                                    category_orders={"pieces_cat": valid_piece_cats_apt}, hover_data=['count'], template="plotly_white"
                                )
                                st.plotly_chart(fig_avg_scatter_pm2_apt, use_container_width=True, key='t2_avg_scatter_apt_pm2')
                        except Exception as e_avg_plot_apt:
                            st.error(f"Erreur affichage nuages moyens (Appt): {e_avg_plot_apt}")
                            logging.error(f"Err plot avg scatter Apt T2: {e_avg_plot_apt}", exc_info=True)
                    else:
                        st.info("Impossible de calculer/afficher des moyennes pour les nuages de points des appartements.")
                    # --- STOCKAGE pour Export Zip ---
                    st.session_state['export_fig_avg_scatter_vf_apt_t2'] = fig_avg_scatter_vf_apt
                    st.session_state['export_fig_avg_scatter_pm2_apt_t2'] = fig_avg_scatter_pm2_apt
                    # --- FIN AJOUT : Nuages de Points MOYENS - Appartements ---



                    # --- DANS ONGLET 2 ---
                    # ... (après le code pour fig_house_scatter_vf et fig_house_scatter_pm2) ...

                    # --- AJOUT : Nuages de Points MOYENS - Maisons ---
                    st.subheader("Surface vs Prix MOYENS - Maisons")
                    st.caption("Ces graphiques montrent la moyenne des prix pour les maisons ayant la même surface et catégorie de pièces.")
                    df_avg_scatter_house = pd.DataFrame()
                    avg_scatter_house_calc_ok = False
                    # Utiliser houses_filtered_scatter qui contient déjà 'pieces_cat'
                    if 'houses_filtered_scatter' in locals() and not houses_filtered_scatter.empty:
                        try:
                            df_avg_scatter_house = houses_filtered_scatter.groupby(['surface_reelle_bati', 'pieces_cat'], observed=True).agg(
                                valeur_fonciere_moyenne=('valeur_fonciere', 'mean'),
                                prix_m2_moyen=('prix_m2', 'mean'),
                                count=('valeur_fonciere', 'size')
                            ).reset_index()
                            if not df_avg_scatter_house.empty:
                                avg_scatter_house_calc_ok = True
                        except Exception as e_avg_calc_house:
                            st.error(f"Erreur calcul moyennes scatter (Maisons): {e_avg_calc_house}")
                            logging.error(f"Err calc avg scatter Maisons T2: {e_avg_calc_house}", exc_info=True)

                    fig_avg_scatter_vf_house = None
                    fig_avg_scatter_pm2_house = None
                    if avg_scatter_house_calc_ok:
                        try:
                            col_avg_scatter_house1, col_avg_scatter_house2 = st.columns(2)
                            # S'assurer que valid_piece_cats_house est défini (il l'est par le bloc scatter précédent)
                            if 'valid_piece_cats_house' not in locals(): valid_piece_cats_house = sorted(df_avg_scatter_house['pieces_cat'].unique())

                            with col_avg_scatter_house1:
                                fig_avg_scatter_vf_house = px.scatter(
                                    df_avg_scatter_house, x="surface_reelle_bati", y="valeur_fonciere_moyenne",
                                    color="pieces_cat", title="Surface vs VF MOYENNE",
                                    labels={"valeur_fonciere_moyenne": "Valeur Moyenne (€)", "surface_reelle_bati": "Surface (m²)", "pieces_cat": "Nb Pièces"},
                                    category_orders={"pieces_cat": valid_piece_cats_house}, hover_data=['count'], template="plotly_white"
                                )
                                st.plotly_chart(fig_avg_scatter_vf_house, use_container_width=True, key='t2_avg_scatter_house_vf')
                            with col_avg_scatter_house2:
                                fig_avg_scatter_pm2_house = px.scatter(
                                    df_avg_scatter_house, x="surface_reelle_bati", y="prix_m2_moyen",
                                    color="pieces_cat", title="Surface vs Prix m² MOYEN",
                                    labels={"prix_m2_moyen": "Prix m² Moyen (€/m²)", "surface_reelle_bati": "Surface (m²)", "pieces_cat": "Nb Pièces"},
                                    category_orders={"pieces_cat": valid_piece_cats_house}, hover_data=['count'], template="plotly_white"
                                )
                                st.plotly_chart(fig_avg_scatter_pm2_house, use_container_width=True, key='t2_avg_scatter_house_pm2')
                        except Exception as e_avg_plot_house:
                            st.error(f"Erreur affichage nuages moyens (Maisons): {e_avg_plot_house}")
                            logging.error(f"Err plot avg scatter Maisons T2: {e_avg_plot_house}", exc_info=True)
                    else:
                         st.info("Impossible de calculer/afficher des moyennes pour les nuages de points des maisons.")
                    # --- STOCKAGE pour Export Zip ---
                    st.session_state['export_fig_avg_scatter_vf_house_t2'] = fig_avg_scatter_vf_house
                    st.session_state['export_fig_avg_scatter_pm2_house_t2'] = fig_avg_scatter_pm2_house
                    # --- FIN AJOUT : Nuages de Points MOYENS - Maisons ---







                    # --- Fin des Graphiques ---

                else: # Si nombre_biens_tab2 == 0
                    st.info("Aucune donnée DVF à afficher pour cet onglet avec les filtres actuels.")
                    # Réinitialiser les clés d'export pour cet onglet
                    keys_to_reset_t2 = [k for k in export_keys if k.endswith('_t2')]
                    for key in keys_to_reset_t2: st.session_state[key] = None

            # --- Fin de with tab2: ---


              # ======================================================
            # ========= ONGLET 3: Comparaison Marché ========= (Base temp.txt + Func Def + Stockage Zip PNG)
            # ======================================================
            with tab3:
                st.header("⚖️ Comparaison Marché Actuel vs Vendu")

                # --- 1. Chargement des données 'À Vendre' ---
                st.subheader("1. Charger les données 'À Vendre' (fichier unique)")
                uploaded_file_moteurimmo_tab3 = st.file_uploader(
                    "Fichier MoteurImmo (Ancien ET Neuf)",
                    type=["csv"],
                    key='upload_moteurimmo_tab3'
                )

                # --- Logique de chargement/nettoyage/split ---
                if uploaded_file_moteurimmo_tab3 is not None:
                    if st.session_state.get('last_upload_moteurimmo_tab3') != uploaded_file_moteurimmo_tab3.name:
                        with st.spinner("Chargement et traitement Fichier MoteurImmo..."):
                            if 'load_clean_moteurimmo' in locals() and callable(load_clean_moteurimmo):
                                df_mi_processed = load_clean_moteurimmo(uploaded_file_moteurimmo_tab3)
                                if not df_mi_processed.empty and 'statut_bien' in df_mi_processed.columns:
                                    df_ancien_split = df_mi_processed[df_mi_processed['statut_bien'] == 'Ancien'].copy()
                                    df_neuf_split = df_mi_processed[df_mi_processed['statut_bien'] == 'Neuf'].copy()
                                    st.session_state['df_ancien_tab3_raw'] = df_ancien_split
                                    st.session_state['df_neuf_tab3_raw'] = df_neuf_split
                                    logging.info(f"Données MoteurImmo splittées: {len(df_ancien_split)} Ancien, {len(df_neuf_split)} Neuf.")
                                    st.success(f"Fichier MoteurImmo traité: {len(df_ancien_split)} biens 'Ancien', {len(df_neuf_split)} biens 'Neuf'.")
                                else:
                                    st.session_state['df_ancien_tab3_raw'] = pd.DataFrame()
                                    st.session_state['df_neuf_tab3_raw'] = pd.DataFrame()
                                    st.warning("Impossible de splitter MoteurImmo (fichier vide, col 'Options' manquante ou erreur).")
                                st.session_state['last_upload_moteurimmo_tab3'] = uploaded_file_moteurimmo_tab3.name
                            else:
                                st.error("Fonction 'load_clean_moteurimmo' non définie.")
                                st.session_state['df_ancien_tab3_raw'] = pd.DataFrame()
                                st.session_state['df_neuf_tab3_raw'] = pd.DataFrame()

                # Récupérer les dataframes depuis session_state
                df_ancien_sale = st.session_state.get('df_ancien_tab3_raw', pd.DataFrame())
                df_neuf_sale = st.session_state.get('df_neuf_tab3_raw', pd.DataFrame())
                # Assurez-vous que df_final_geo_filtered est bien défini
                df_dvf_sold = df_final_geo_filtered if 'df_final_geo_filtered' in locals() else pd.DataFrame()
                has_dvf = not df_dvf_sold.empty
                has_ancien = not df_ancien_sale.empty
                has_neuf = not df_neuf_sale.empty

                # --- Section Comparaison ---
                keys_to_reset_t3 = [k for k in export_keys if k.endswith('_t3')] # Utilise la liste globale export_keys
                if not has_dvf or not (has_ancien or has_neuf):
                    st.warning("Chargez le fichier DVF via la sidebar ET le fichier MoteurImmo ci-dessus pour activer la comparaison.")
                    for key in keys_to_reset_t3: st.session_state[key] = None
                else:
                    st.markdown("---")
                    st.subheader("2. Affiner les données pour la comparaison")

                    # --- Filtres Communes Tab 3 ('A Vendre') ---
                    # ... (Code Filtres Communes inchangé par rapport à temp.txt) ...
                    st.markdown("**Filtres Communes (Marché 'À Vendre')**")
                    col_com_an, col_com_ne = st.columns(2)
                    with col_com_an:
                        communes_dispo_ancien = sorted(df_ancien_sale['nom_commune'].unique()) if has_ancien and 'nom_commune' in df_ancien_sale.columns else []
                        key_communes_ancien = 't3_communes_ancien'
                        default_communes_ancien = st.session_state.get(key_communes_ancien, communes_dispo_ancien)
                        validated_communes_ancien = [c for c in default_communes_ancien if c in communes_dispo_ancien]
                        if not validated_communes_ancien and communes_dispo_ancien: validated_communes_ancien = communes_dispo_ancien
                        if validated_communes_ancien != st.session_state.get(key_communes_ancien): st.session_state[key_communes_ancien] = validated_communes_ancien
                        selected_communes_ancien_tab3 = st.multiselect("Communes Ancien :", communes_dispo_ancien, default=st.session_state.get(key_communes_ancien, []), key=key_communes_ancien+"_widget", disabled=not has_ancien)
                    with col_com_ne:
                        communes_dispo_neuf = sorted(df_neuf_sale['nom_commune'].unique()) if has_neuf and 'nom_commune' in df_neuf_sale.columns else []
                        key_communes_neuf = 't3_communes_neuf'
                        default_communes_neuf = st.session_state.get(key_communes_neuf, communes_dispo_neuf)
                        validated_communes_neuf = [c for c in default_communes_neuf if c in communes_dispo_neuf]
                        if not validated_communes_neuf and communes_dispo_neuf: validated_communes_neuf = communes_dispo_neuf
                        if validated_communes_neuf != st.session_state.get(key_communes_neuf): st.session_state[key_communes_neuf] = validated_communes_neuf
                        selected_communes_neuf_tab3 = st.multiselect("Communes Neuf :", communes_dispo_neuf, default=st.session_state.get(key_communes_neuf, []), key=key_communes_neuf+"_widget", disabled=not has_neuf)

                    df_ancien_commune_filtered = df_ancien_sale[df_ancien_sale['nom_commune'].isin(selected_communes_ancien_tab3)].copy() if has_ancien and selected_communes_ancien_tab3 else pd.DataFrame()
                    df_neuf_commune_filtered = df_neuf_sale[df_neuf_sale['nom_commune'].isin(selected_communes_neuf_tab3)].copy() if has_neuf and selected_communes_neuf_tab3 else pd.DataFrame()
                    has_ancien_comm_filtered = not df_ancien_commune_filtered.empty
                    has_neuf_comm_filtered = not df_neuf_commune_filtered.empty

                    # --- Autres Filtres ('A Vendre' - Version temp.txt avec Number Inputs & Sliders) ---
                    col_f_ancien, col_f_neuf = st.columns(2)
                    DEFAULT_RANGE_T3 = (0, 1) # Range par défaut pour Slider Pièces

                    # Initialiser variables de sélection
                    selected_type_ancien, selected_pieces_ancien = [], DEFAULT_RANGE_T3
                    selected_surf_ancien_min, selected_surf_ancien_max = 0, 10000
                    selected_vf_ancien_min, selected_vf_ancien_max = 0, 10000000
                    selected_prix_ancien_min, selected_prix_ancien_max = 0, 50000

                    selected_type_neuf, selected_pieces_neuf = [], DEFAULT_RANGE_T3
                    selected_surf_neuf_min, selected_surf_neuf_max = 0, 10000
                    selected_vf_neuf_min, selected_vf_neuf_max = 0, 10000000
                    selected_prix_neuf_min, selected_prix_neuf_max = 0, 50000

                    # --- Colonne Filtres ANCIEN (Logique Number Input / Slider temp.txt) ---
                    with col_f_ancien:
                        # ... (Code des filtres Ancien : Type, Surface, VF, Prix/m2, Pieces - INCHANGÉ par rapport à temp.txt) ...
                        st.markdown("**Filtres Biens Anciens (À Vendre)**")
                        if has_ancien_comm_filtered:
                            df_current_ancien = df_ancien_commune_filtered
                            # Filtre Type (Multiselect)
                            key_type_ancien = "t3_type_ancien"; types_ancien = sorted(df_current_ancien['type_local'].unique()) if 'type_local' in df_current_ancien.columns else []
                            if types_ancien:
                                default_type_ancien = st.session_state.get(key_type_ancien, types_ancien); default_type_ancien = [t for t in default_type_ancien if t in types_ancien]
                                if not default_type_ancien: default_type_ancien = types_ancien
                                if default_type_ancien != st.session_state.get(key_type_ancien): st.session_state[key_type_ancien] = default_type_ancien
                                selected_type_ancien = st.multiselect("Type Bien Ancien", types_ancien, default=st.session_state.get(key_type_ancien, []), key=key_type_ancien+"_widget")
                            else: selected_type_ancien = []; st.caption("Pas de types.")
                            st.markdown("---")

                            # Filtre Surface (Number Inputs)
                            filter_key_surf_anc = "t3_surf_ancien"; min_key_s_anc = f"{filter_key_surf_anc}_min"; max_key_s_anc = f"{filter_key_surf_anc}_max"
                            col_surf = 'surface_reelle_bati'; min_s_data, max_s_data = 0, 10000
                            if col_surf in df_current_ancien.columns and not df_current_ancien[col_surf].dropna().empty:
                                surf_valides = df_current_ancien[df_current_ancien[col_surf] > 0][col_surf].dropna()
                                if not surf_valides.empty: min_s_data, max_s_data = int(surf_valides.min()), int(surf_valides.max())
                            default_min_s = st.session_state.get(min_key_s_anc, min_s_data); default_max_s = st.session_state.get(max_key_s_anc, max_s_data)
                            default_min_s = max(min_s_data, default_min_s); default_max_s = min(max_s_data, default_max_s);
                            if default_max_s < default_min_s: default_max_s = default_min_s
                            st.write("Surface Ancien (m²):"); s_col_min_anc, s_col_max_anc = st.columns(2)
                            with s_col_min_anc: selected_surf_ancien_min = st.number_input("Min", min_value=min_s_data, max_value=max_s_data, value=default_min_s, step=5, key=min_key_s_anc, format="%d")
                            with s_col_max_anc: min_val_for_max = selected_surf_ancien_min if selected_surf_ancien_min is not None else min_s_data; selected_surf_ancien_max = st.number_input("Max", min_value=min_val_for_max, max_value=max_s_data, value=max(min_val_for_max, default_max_s), step=5, key=max_key_s_anc, format="%d")
                            st.markdown("---")

                            # Filtre Valeur Foncière (Number Inputs)
                            filter_key_vf_anc = "t3_vf_ancien"; min_key_vf_anc = f"{filter_key_vf_anc}_min"; max_key_vf_anc = f"{filter_key_vf_anc}_max"
                            col_vf = 'valeur_fonciere'; min_vf_data, max_vf_data = 0, 10000000
                            if col_vf in df_current_ancien.columns and not df_current_ancien[col_vf].dropna().empty:
                                vf_valides = df_current_ancien[df_current_ancien[col_vf] > 0][col_vf].dropna()
                                if not vf_valides.empty: min_vf_data, max_vf_data = int(vf_valides.min()), int(vf_valides.max())
                            default_min_vf = st.session_state.get(min_key_vf_anc, min_vf_data); default_max_vf = st.session_state.get(max_key_vf_anc, max_vf_data)
                            default_min_vf = max(min_vf_data, default_min_vf); default_max_vf = min(max_vf_data, default_max_vf)
                            if default_max_vf < default_min_vf: default_max_vf = default_min_vf
                            st.write("Valeur Foncière Ancien (€):"); vf_col_min_anc, vf_col_max_anc = st.columns(2)
                            with vf_col_min_anc: selected_vf_ancien_min = st.number_input("Min VF Anc.", min_value=min_vf_data, max_value=max_vf_data, value=default_min_vf, step=10000, key=min_key_vf_anc, format="%d", label_visibility="collapsed")
                            with vf_col_max_anc: min_val_for_max_vf = selected_vf_ancien_min if selected_vf_ancien_min is not None else min_vf_data; selected_vf_ancien_max = st.number_input("Max VF Anc.", min_value=min_val_for_max_vf, max_value=max_vf_data, value=max(min_val_for_max_vf, default_max_vf), step=10000, key=max_key_vf_anc, format="%d", label_visibility="collapsed")
                            st.markdown("---")

                            # Filtre Prix/m² (Number Inputs)
                            filter_key_prix_anc = "t3_prix_ancien"; min_key_p_anc = f"{filter_key_prix_anc}_min"; max_key_p_anc = f"{filter_key_prix_anc}_max"
                            col_prix = 'prix_m2'; min_p_data, max_p_data = 0, 50000
                            if col_prix in df_current_ancien.columns and not df_current_ancien[col_prix].dropna().empty:
                                prix_valides = df_current_ancien[df_current_ancien[col_prix] > 0][col_prix].dropna()
                                if not prix_valides.empty: min_p_data, max_p_data = int(prix_valides.min()), int(prix_valides.max())
                            default_min_p = st.session_state.get(min_key_p_anc, min_p_data); default_max_p = st.session_state.get(max_key_p_anc, max_p_data)
                            default_min_p = max(min_p_data, default_min_p); default_max_p = min(max_p_data, default_max_p)
                            if default_max_p < default_min_p: default_max_p = default_min_p
                            st.write("Prix/m² Ancien (€/m²):"); p_col_min_anc, p_col_max_anc = st.columns(2)
                            with p_col_min_anc: selected_prix_ancien_min = st.number_input("Min", min_value=min_p_data, max_value=max_p_data, value=default_min_p, step=100, key=min_key_p_anc, format="%d")
                            with p_col_max_anc: min_val_for_max_p = selected_prix_ancien_min if selected_prix_ancien_min is not None else min_p_data; selected_prix_ancien_max = st.number_input("Max", min_value=min_val_for_max_p, max_value=max_p_data, value=max(min_val_for_max_p, default_max_p), step=100, key=max_key_p_anc, format="%d")
                            st.markdown("---")

                            # Filtre Nb Pièces (Slider)
                            key_pieces_ancien = "t3_pieces_ancien"; col_pieces = 'nombre_pieces_principales'
                            min_sl_pc_anc, max_sl_pc_anc, calc_def_pc_anc = DEFAULT_RANGE_T3[0], DEFAULT_RANGE_T3[1]+1, DEFAULT_RANGE_T3
                            if col_pieces in df_current_ancien.columns and not df_current_ancien[col_pieces].dropna().empty:
                                pieces_valides = pd.to_numeric(df_current_ancien[col_pieces], errors='coerce').dropna().pipe(lambda s: s[s >= 0])
                                if not pieces_valides.empty: min_sl_pc_anc, max_sl_pc_anc = int(pieces_valides.min()), int(pieces_valides.max()); calc_def_pc_anc = (min_sl_pc_anc, max_sl_pc_anc)
                                if min_sl_pc_anc == max_sl_pc_anc: max_sl_pc_anc += 1

                            # --- Correction pour éviter l'avertissement ---
                            # 1. Lire l'état actuel ou utiliser le défaut calculé
                            current_value_or_default_pc_anc = st.session_state.get(key_pieces_ancien, calc_def_pc_anc)

                            # 2. Borner la valeur
                            bound_v_pc_anc = (
                                max(min_sl_pc_anc, current_value_or_default_pc_anc[0]),
                                min(max_sl_pc_anc, current_value_or_default_pc_anc[1])
                            )
                            if bound_v_pc_anc[1] < bound_v_pc_anc[0]:
                                bound_v_pc_anc = (bound_v_pc_anc[0], bound_v_pc_anc[0])

                            # 3. Créer le slider
                            selected_pieces_ancien = st.slider(
                                "Nb Pièces Ancien",
                                min_value=min_sl_pc_anc,
                                max_value=max_sl_pc_anc,
                                value=bound_v_pc_anc, # Utiliser la valeur bornée
                                key=key_pieces_ancien      # Lier à l'état
                            )
                            # --- Fin Correction ---


                            # if key_pieces_ancien not in st.session_state: st.session_state[key_pieces_ancien] = calc_def_pc_anc
                            # curr_v_pc_anc = st.session_state.get(key_pieces_ancien, calc_def_pc_anc); bound_v_pc_anc = (max(min_sl_pc_anc, curr_v_pc_anc[0]), min(max_sl_pc_anc, curr_v_pc_anc[1]))
                            # if bound_v_pc_anc[1] < bound_v_pc_anc[0]: bound_v_pc_anc = (bound_v_pc_anc[0], bound_v_pc_anc[0])
                            # selected_pieces_ancien = st.slider("Nb Pièces Ancien", min_value=min_sl_pc_anc, max_value=max_sl_pc_anc, value=bound_v_pc_anc, key=key_pieces_ancien)
                       

                        else:
                            st.info("Pas de données 'Ancien' pour les communes sélectionnées.")

                    # --- Colonne Filtres NEUF (Logique Number Input / Slider temp.txt) ---
                    with col_f_neuf:
                        st.markdown("**Filtres Biens Neufs (À Vendre)**")
                        if has_neuf_comm_filtered:
                            df_current_neuf = df_neuf_commune_filtered
                            # Filtre Type (Multiselect)
                            key_type_neuf = "t3_type_neuf"; types_neuf = sorted(df_current_neuf['type_local'].unique()) if 'type_local' in df_current_neuf.columns else []
                            if types_neuf:
                                default_type_neuf = st.session_state.get(key_type_neuf, types_neuf); default_type_neuf = [t for t in default_type_neuf if t in types_neuf]
                                if not default_type_neuf: default_type_neuf = types_neuf
                                if default_type_neuf != st.session_state.get(key_type_neuf): st.session_state[key_type_neuf] = default_type_neuf
                                selected_type_neuf = st.multiselect("Type Bien Neuf", types_neuf, default=st.session_state.get(key_type_neuf, []), key=key_type_neuf+"_widget")
                            else: selected_type_neuf = []; st.caption("Pas de types.")
                            st.markdown("---")

                            # Filtre Surface (Number Inputs)
                            filter_key_surf_neuf = "t3_surf_neuf"; min_key_sn = f"{filter_key_surf_neuf}_min"; max_key_sn = f"{filter_key_surf_neuf}_max"
                            col_surf_n = 'surface_reelle_bati'; min_sn_data, max_sn_data = 0, 10000
                            if col_surf_n in df_current_neuf.columns and not df_current_neuf[col_surf_n].dropna().empty:
                                surf_valides_n = df_current_neuf[df_current_neuf[col_surf_n] > 0][col_surf_n].dropna()
                                if not surf_valides_n.empty: min_sn_data, max_sn_data = int(surf_valides_n.min()), int(surf_valides_n.max())
                            default_min_sn = st.session_state.get(min_key_sn, min_sn_data); default_max_sn = st.session_state.get(max_key_sn, max_sn_data)
                            default_min_sn = max(min_sn_data, default_min_sn); default_max_sn = min(max_sn_data, default_max_sn)
                            if default_max_sn < default_min_sn: default_max_sn = default_min_sn
                            st.write("Surface Neuf (m²):"); sn_col_min, sn_col_max = st.columns(2)
                            with sn_col_min: selected_surf_neuf_min = st.number_input("Min", min_value=min_sn_data, max_value=max_sn_data, value=default_min_sn, step=5, key=min_key_sn, format="%d")
                            with sn_col_max: min_val_for_max_sn = selected_surf_neuf_min if selected_surf_neuf_min is not None else min_sn_data; selected_surf_neuf_max = st.number_input("Max", min_value=min_val_for_max_sn, max_value=max_sn_data, value=max(min_val_for_max_sn, default_max_sn), step=5, key=max_key_sn, format="%d")
                            st.markdown("---")

                            # Filtre Valeur Foncière (Number Inputs)
                            filter_key_vf_neuf = "t3_vf_neuf"; min_key_vf_neuf = f"{filter_key_vf_neuf}_min"; max_key_vf_neuf = f"{filter_key_vf_neuf}_max"
                            col_vf_n = 'valeur_fonciere'; min_vf_n_data, max_vf_n_data = 0, 10000000
                            if col_vf_n in df_current_neuf.columns and not df_current_neuf[col_vf_n].dropna().empty:
                                vf_valides_n = df_current_neuf[df_current_neuf[col_vf_n] > 0][col_vf_n].dropna()
                                if not vf_valides_n.empty: min_vf_n_data, max_vf_n_data = int(vf_valides_n.min()), int(vf_valides_n.max())
                            default_min_vf_n = st.session_state.get(min_key_vf_neuf, min_vf_n_data); default_max_vf_n = st.session_state.get(max_key_vf_neuf, max_vf_n_data)
                            default_min_vf_n = max(min_vf_n_data, default_min_vf_n); default_max_vf_n = min(max_vf_n_data, default_max_vf_n)
                            if default_max_vf_n < default_min_vf_n: default_max_vf_n = default_min_vf_n
                            st.write("Valeur Foncière Neuf (€):"); vf_col_min_neuf, vf_col_max_neuf = st.columns(2)
                            with vf_col_min_neuf: selected_vf_neuf_min = st.number_input("Min VF Neuf", min_value=min_vf_n_data, max_value=max_vf_n_data, value=default_min_vf_n, step=10000, key=min_key_vf_neuf, format="%d", label_visibility="collapsed")
                            with vf_col_max_neuf: min_val_for_max_vf_n = selected_vf_neuf_min if selected_vf_neuf_min is not None else min_vf_n_data; selected_vf_neuf_max = st.number_input("Max VF Neuf", min_value=min_val_for_max_vf_n, max_value=max_vf_n_data, value=max(min_val_for_max_vf_n, default_max_vf_n), step=10000, key=max_key_vf_neuf, format="%d", label_visibility="collapsed")
                            st.markdown("---")

                            # Filtre Prix/m² (Number Inputs)
                            filter_key_prix_neuf = "t3_prix_neuf"; min_key_pn = f"{filter_key_prix_neuf}_min"; max_key_pn = f"{filter_key_prix_neuf}_max"
                            col_prix_n = 'prix_m2'; min_pn_data, max_pn_data = 0, 50000
                            if col_prix_n in df_current_neuf.columns and not df_current_neuf[col_prix_n].dropna().empty:
                                prix_valides_n = df_current_neuf[df_current_neuf[col_prix_n] > 0][col_prix_n].dropna()
                                if not prix_valides_n.empty: min_pn_data, max_pn_data = int(prix_valides_n.min()), int(prix_valides_n.max())
                            default_min_pn = st.session_state.get(min_key_pn, min_pn_data); default_max_pn = st.session_state.get(max_key_pn, max_pn_data)
                            default_min_pn = max(min_pn_data, default_min_pn); default_max_pn = min(max_pn_data, default_max_pn)
                            if default_max_pn < default_min_pn: default_max_pn = default_min_pn
                            st.write("Prix/m² Neuf (€/m²):"); pn_col_min, pn_col_max = st.columns(2)
                            with pn_col_min: selected_prix_neuf_min = st.number_input("Min", min_value=min_pn_data, max_value=max_pn_data, value=default_min_pn, step=100, key=min_key_pn, format="%d")
                            with pn_col_max: min_val_for_max_pn = selected_prix_neuf_min if selected_prix_neuf_min is not None else min_pn_data; selected_prix_neuf_max = st.number_input("Max", min_value=min_val_for_max_pn, max_value=max_pn_data, value=max(min_val_for_max_pn, default_max_pn), step=100, key=max_key_pn, format="%d")
                            st.markdown("---")

                            # Filtre Nb Pièces (Slider)
                            key_pieces_neuf = "t3_pieces_neuf"; col_pieces_n = 'nombre_pieces_principales'
                            min_sl_n, max_sl_n, calc_def_pn = DEFAULT_RANGE_T3[0], DEFAULT_RANGE_T3[1]+1, DEFAULT_RANGE_T3
                            if col_pieces_n in df_current_neuf.columns and not df_current_neuf[col_pieces_n].dropna().empty:
                                 pieces_valides_n = pd.to_numeric(df_current_neuf[col_pieces_n], errors='coerce').dropna().pipe(lambda s: s[s >= 0])
                                 if not pieces_valides_n.empty: min_sl_n, max_sl_n = int(pieces_valides_n.min()), int(pieces_valides_n.max()); calc_def_pn = (min_sl_n, max_sl_n)
                                 if min_sl_n == max_sl_n: max_sl_n += 1

                           # --- Correction pour éviter l'avertissement ---
                            # 1. Lire l'état actuel ou utiliser le défaut calculé
                            current_value_or_default_pn = st.session_state.get(key_pieces_neuf, calc_def_pn)

                            # 2. Borner la valeur
                            bound_v_n = (
                                max(min_sl_n, current_value_or_default_pn[0]),
                                min(max_sl_n, current_value_or_default_pn[1])
                            )
                            if bound_v_n[1] < bound_v_n[0]:
                                bound_v_n = (bound_v_n[0], bound_v_n[0])

                            # 3. Créer le slider
                            selected_pieces_neuf = st.slider(
                                "Nb Pièces Neuf",
                                min_value=min_sl_n,
                                max_value=max_sl_n,
                                value=bound_v_n, # Utiliser la valeur bornée
                                key=key_pieces_neuf      # Lier à l'état
                            )
                            # --- Fin Correction ---



                            # if key_pieces_neuf not in st.session_state: st.session_state[key_pieces_neuf] = calc_def_pn
                            # curr_v_n = st.session_state.get(key_pieces_neuf, calc_def_pn); bound_v_n = (max(min_sl_n, curr_v_n[0]), min(max_sl_n, curr_v_n[1]))
                            # if bound_v_n[1] < bound_v_n[0]: bound_v_n = (bound_v_n[0], bound_v_n[0])
                            # selected_pieces_neuf = st.slider("Nb Pièces Neuf", min_value=min_sl_n, max_value=max_sl_n, value=bound_v_n, key=key_pieces_neuf)

                        else:
                            st.info("Pas de données 'Neuf' pour les communes sélectionnées.")

                    # --- Application Finale des Filtres ('A Vendre') ---
                    df_ancien_filtered_tab3 = pd.DataFrame() # Init
                    if has_ancien_comm_filtered:
                        try:
                            # Utiliser les variables des Number Inputs et Sliders
                            mask_ancien = pd.Series(True, index=df_ancien_commune_filtered.index)
                            if 'type_local' in df_ancien_commune_filtered.columns and selected_type_ancien: mask_ancien &= df_ancien_commune_filtered['type_local'].isin(selected_type_ancien)
                            if 'surface_reelle_bati' in df_ancien_commune_filtered.columns: mask_ancien &= df_ancien_commune_filtered['surface_reelle_bati'].between(selected_surf_ancien_min, selected_surf_ancien_max)
                            if 'valeur_fonciere' in df_ancien_commune_filtered.columns: mask_ancien &= df_ancien_commune_filtered['valeur_fonciere'].between(selected_vf_ancien_min, selected_vf_ancien_max)
                            if 'prix_m2' in df_ancien_commune_filtered.columns: mask_ancien &= df_ancien_commune_filtered['prix_m2'].between(selected_prix_ancien_min, selected_prix_ancien_max)
                            if 'nombre_pieces_principales' in df_ancien_commune_filtered.columns: mask_ancien &= df_ancien_commune_filtered['nombre_pieces_principales'].between(selected_pieces_ancien[0], selected_pieces_ancien[1])
                            df_ancien_filtered_tab3 = df_ancien_commune_filtered[mask_ancien].copy()
                        except Exception as e: st.error(f"Erreur filtrage final Ancien: {e}"); logging.error(f"Err Ancien T3: {e}", exc_info=True)

                    df_neuf_filtered_tab3 = pd.DataFrame() # Init
                    if has_neuf_comm_filtered:
                        try:
                            # Utiliser les variables des Number Inputs et Sliders
                            mask_neuf = pd.Series(True, index=df_neuf_commune_filtered.index)
                            if 'type_local' in df_neuf_commune_filtered.columns and selected_type_neuf: mask_neuf &= df_neuf_commune_filtered['type_local'].isin(selected_type_neuf)
                            if 'surface_reelle_bati' in df_neuf_commune_filtered.columns: mask_neuf &= df_neuf_commune_filtered['surface_reelle_bati'].between(selected_surf_neuf_min, selected_surf_neuf_max)
                            if 'valeur_fonciere' in df_neuf_commune_filtered.columns: mask_neuf &= df_neuf_commune_filtered['valeur_fonciere'].between(selected_vf_neuf_min, selected_vf_neuf_max)
                            if 'prix_m2' in df_neuf_commune_filtered.columns: mask_neuf &= df_neuf_commune_filtered['prix_m2'].between(selected_prix_neuf_min, selected_prix_neuf_max)
                            if 'nombre_pieces_principales' in df_neuf_commune_filtered.columns: mask_neuf &= df_neuf_commune_filtered['nombre_pieces_principales'].between(selected_pieces_neuf[0], selected_pieces_neuf[1])
                            df_neuf_filtered_tab3 = df_neuf_commune_filtered[mask_neuf].copy()
                        except Exception as e: st.error(f"Erreur filtrage final Neuf: {e}"); logging.error(f"Err Neuf T3: {e}", exc_info=True)

                    # Stockage session state des DFs filtrés finaux de Tab 3 (pour Tab 6)
                    st.session_state['filtered_df_tab3_ancien'] = df_ancien_filtered_tab3.copy()
                    st.session_state['filtered_df_tab3_neuf'] = df_neuf_filtered_tab3.copy()
                    logging.info(f"Tab 3: DF Ancien filtré stocké ({len(df_ancien_filtered_tab3)} lignes).")
                    logging.info(f"Tab 3: DF Neuf filtré stocké ({len(df_neuf_filtered_tab3)} lignes).")

                    # --- 3. Comparaison des Marchés Filtrés ---
                    st.markdown("---")
                    st.subheader("Comparaison des Marchés Filtrés")

                    # --- Application filtres Tab 1 aux données DVF ---
                    df_dvf_filtered_like_tab1 = pd.DataFrame()
                    df_dvf_compare = pd.DataFrame() # Init df_dvf_compare
                    if 'df_dvf_sold' in locals() and not df_dvf_sold.empty:
                        base_dvf_for_compare = df_dvf_sold.copy()
                        st.write("Application des filtres du Tableau de Bord (Onglet 1) aux données DVF...")
                        try:
                            # Récupérer les valeurs des filtres de Tab 1 depuis session_state
                            t1_annees = st.session_state.get('t1_slider_annee', (0,1))
                            t1_vf_min = st.session_state.get('t1_filter_vf_min', 0)
                            t1_vf_max = st.session_state.get('t1_filter_vf_max', np.inf)
                            t1_pm2_min = st.session_state.get('t1_filter_pm2_min', 0)
                            t1_pm2_max = st.session_state.get('t1_filter_pm2_max', np.inf)
                            t1_surf_min = st.session_state.get('t1_filter_surf_min', 0)
                            t1_surf_max = st.session_state.get('t1_filter_surf_max', np.inf)
                            t1_pieces = st.session_state.get('t1_slider_pieces', (0,100))
                            t1_terrain = st.session_state.get('t1_radio_terrain', 'Tous')
                            t1_types = st.session_state.get('t1_multi_type', [])

                            # Préparer le masque
                            mask_dvf_t1 = pd.Series(True, index=base_dvf_for_compare.index)
                            if 'annee' in base_dvf_for_compare.columns: mask_dvf_t1 &= base_dvf_for_compare['annee'].between(t1_annees[0], t1_annees[1])
                            if 'valeur_fonciere' in base_dvf_for_compare.columns: mask_dvf_t1 &= base_dvf_for_compare['valeur_fonciere'].between(t1_vf_min, t1_vf_max)
                            if 'prix_m2' in base_dvf_for_compare.columns: mask_dvf_t1 &= base_dvf_for_compare['prix_m2'].between(t1_pm2_min, t1_pm2_max)
                            if 'surface_reelle_bati' in base_dvf_for_compare.columns: mask_dvf_t1 &= base_dvf_for_compare['surface_reelle_bati'].between(t1_surf_min, t1_surf_max)
                            if 'nombre_pieces_principales' in base_dvf_for_compare.columns: mask_dvf_t1 &= base_dvf_for_compare['nombre_pieces_principales'].between(t1_pieces[0], t1_pieces[1])
                            if 'type_local' in base_dvf_for_compare.columns and t1_types: mask_dvf_t1 &= base_dvf_for_compare['type_local'].isin(t1_types)

                            df_dvf_filtered_like_tab1 = base_dvf_for_compare[mask_dvf_t1].copy()

                            if 'surface_terrain' in df_dvf_filtered_like_tab1.columns:
                                if t1_terrain == 'Oui': df_dvf_filtered_like_tab1 = df_dvf_filtered_like_tab1[df_dvf_filtered_like_tab1['surface_terrain'].fillna(0) > 0]
                                elif t1_terrain == 'Non': df_dvf_filtered_like_tab1 = df_dvf_filtered_like_tab1[df_dvf_filtered_like_tab1['surface_terrain'].fillna(0) == 0]

                            if df_dvf_filtered_like_tab1.empty: st.warning("Aucune donnée DVF ne correspond aux filtres de l'Onglet 1.")
                            else: st.info(f"{len(df_dvf_filtered_like_tab1)} biens DVF correspondent aux filtres de l'Onglet 1.")

                        except KeyError as e_key: st.warning(f"Filtre Tab 1 manquant ({e_key}).")
                        except Exception as e_filter_dvf: st.error(f"Erreur application filtres T1 sur DVF: {e_filter_dvf}"); df_dvf_filtered_like_tab1 = base_dvf_for_compare
                    else:
                        st.warning("Données DVF (vendus) non disponibles pour comparaison.")

                    # Assigner le résultat
                    df_dvf_compare = df_dvf_filtered_like_tab1

                    # --- AJOUT: Définition de calculate_comparative_stats ICI ---
                    # (Ou assurez-vous qu'elle est définie globalement avant)
                    def calculate_comparative_stats(df, label):
                        stats = {'Source': label, 'Nb Biens': 0, 'Prix/m² Moyen': 'N/A', 'Prix/m² Median': 'N/A', 'Prix/m² Min': 'N/A', 'Prix/m² Max': 'N/A', 'Prix/m² Moyen Num': np.nan}
                        if df is None or df.empty: return stats
                        stats['Nb Biens'] = len(df)
                        if 'prix_m2' in df.columns:
                            valid_prices = pd.to_numeric(df['prix_m2'], errors='coerce').dropna()
                            valid_prices = valid_prices[valid_prices > 0]
                            valid_prices = valid_prices[np.isfinite(valid_prices)]
                            if not valid_prices.empty:
                                mean_val, median_val, min_val, max_val = valid_prices.mean(), valid_prices.median(), valid_prices.min(), valid_prices.max()
                                def format_stat_comp(value):
                                     if pd.notna(value) and np.isfinite(value):
                                         try: return f"{int(round(value)):,} €/m²".replace(',', ' ')
                                         except: return str(value) + " €/m²"
                                     return 'N/A'
                                stats.update({
                                    'Prix/m² Moyen': format_stat_comp(mean_val), 'Prix/m² Median': format_stat_comp(median_val),
                                    'Prix/m² Min': format_stat_comp(min_val), 'Prix/m² Max': format_stat_comp(max_val),
                                    'Prix/m² Moyen Num': mean_val
                                })
                        return stats
                    # --- FIN Définition ---

                    # --- Tableau Statistique Comparatif ---
                    df_stats_compare_raw = pd.DataFrame() # Init
                    stats_list = []
                    dict_dvf = calculate_comparative_stats(df_dvf_compare, "Vendu (DVF Filtré Onglet1)")
                    dict_ancien = calculate_comparative_stats(df_ancien_filtered_tab3, "À Vendre (Ancien Filtré Onglet3)")
                    dict_neuf = calculate_comparative_stats(df_neuf_filtered_tab3, "À Vendre (Neuf Filtré Onglet3)")
                    stats_list = [d for d in [dict_dvf, dict_ancien, dict_neuf] if d and d.get('Nb Biens', 0) > 0]

                    if len(stats_list) > 0:
                        df_stats_compare_raw = pd.DataFrame(stats_list).set_index('Source')
                        # Calcul Diff %
                        df_stats_compare_raw['Diff % Moyenne vs Vendu'] = 'N/A'
                        dvf_avg_price_num = df_stats_compare_raw.loc["Vendu (DVF Filtré Onglet1)", 'Prix/m² Moyen Num'] if "Vendu (DVF Filtré Onglet1)" in df_stats_compare_raw.index else np.nan
                        if pd.notna(dvf_avg_price_num) and dvf_avg_price_num > 0:
                            for idx in df_stats_compare_raw.index:
                                if idx != "Vendu (DVF Filtré Onglet1)":
                                    current_avg_num = df_stats_compare_raw.loc[idx, 'Prix/m² Moyen Num']
                                    if pd.notna(current_avg_num):
                                        diff_percent = ((current_avg_num - dvf_avg_price_num) / dvf_avg_price_num) * 100
                                        df_stats_compare_raw.loc[idx, 'Diff % Moyenne vs Vendu'] = f"{diff_percent:+.1f}%"
                        else: st.caption("Calcul Diff % impossible (Moyenne DVF invalide).")

                        # Afficher et Stocker
                        cols_to_display = ['Nb Biens', 'Prix/m² Moyen', 'Prix/m² Median', 'Prix/m² Min', 'Prix/m² Max', 'Diff % Moyenne vs Vendu']
                        cols_final_display = [c for c in cols_to_display if c in df_stats_compare_raw.columns]
                        st.dataframe(df_stats_compare_raw[cols_final_display], hide_index=False) # Afficher l'index (Source)
                        # --- STOCKAGE pour Export Zip ---
                        st.session_state['export_df_stats_compare_t3'] = df_stats_compare_raw[cols_final_display].copy()
                        logging.info(f"Tab 3: df_stats_compare stocké (taille {df_stats_compare_raw.shape}).")
                        # --- FIN STOCKAGE ---

                        if len(stats_list) < 2: st.info("Chargez et filtrez au moins une autre source pour comparaison.")
                    else:
                        st.warning("Aucune donnée statistique à afficher pour la comparaison.")
                        st.session_state['export_df_stats_compare_t3'] = None # Reset

                    # --- Histogrammes Comparatifs ---
                    dfs_to_combine = []
                    fig_hist_compare_pm2 = None; fig_hist_compare_vf = None # Init figures
                    # Assurer que prepare_hist_data est définie
                    if 'prepare_hist_data' in locals() and callable(prepare_hist_data):
                        df_dvf_hist = prepare_hist_data(df_dvf_compare, "Vendu (DVF Filtré Tab 1)")
                        df_ancien_hist = prepare_hist_data(df_ancien_filtered_tab3, "À Vendre (Ancien Filtré Tab 3)")
                        df_neuf_hist = prepare_hist_data(df_neuf_filtered_tab3, "À Vendre (Neuf Filtré Tab 3)")
                        if df_dvf_hist is not None: dfs_to_combine.append(df_dvf_hist)
                        if df_ancien_hist is not None: dfs_to_combine.append(df_ancien_hist)
                        if df_neuf_hist is not None: dfs_to_combine.append(df_neuf_hist)

                        if dfs_to_combine:
                            df_hist_compare = pd.concat(dfs_to_combine, ignore_index=True)
                            st.subheader("Distributions Comparées")
                            col_hist_compare1, col_hist_compare2 = st.columns(2)
                            with col_hist_compare1:
                                st.write("**Par Prix au m²**")
                                try:
                                    fig_hist_compare_pm2 = px.histogram(df_hist_compare, x="prix_m2", color="Source", barmode='group', marginal="box", labels={'prix_m2': 'Prix au m² (€/m²)'}, nbins=50, template="plotly_white") # Ajout template
                                    fig_hist_compare_pm2.update_layout(yaxis_title="Nombre de Biens", title_text="", bargap=0.1)
                                    st.plotly_chart(fig_hist_compare_pm2, use_container_width=True, key="t3_hist_compare_pm2")
                                    # STOCKAGE pour Export Zip
                                    st.session_state['export_fig_hist_compare_pm2_t3'] = fig_hist_compare_pm2
                                except Exception as e: st.error(f"Err hist PM2: {e}"); logging.error(f"Err hist comp pm2 T3: {e}", exc_info=True); st.session_state['export_fig_hist_compare_pm2_t3'] = None
                            with col_hist_compare2:
                                st.write("**Par Valeur Foncière**")
                                try:
                                    fig_hist_compare_vf = px.histogram(df_hist_compare, x="valeur_fonciere", color="Source", barmode='group', marginal="box", labels={'valeur_fonciere': 'Valeur Foncière (€)'}, nbins=50, template="plotly_white") # Ajout template
                                    fig_hist_compare_vf.update_layout(yaxis_title="Nombre de Biens", title_text="", bargap=0.1)
                                    st.plotly_chart(fig_hist_compare_vf, use_container_width=True, key="t3_hist_compare_vf")
                                    # STOCKAGE pour Export Zip
                                    st.session_state['export_fig_hist_compare_vf_t3'] = fig_hist_compare_vf
                                except Exception as e: st.error(f"Err hist VF: {e}"); logging.error(f"Err hist comp vf T3: {e}", exc_info=True); st.session_state['export_fig_hist_compare_vf_t3'] = None
                        else:
                             st.info("Aucune donnée valide pour histogrammes comparatifs.")
                             st.session_state['export_fig_hist_compare_pm2_t3'] = None
                             st.session_state['export_fig_hist_compare_vf_t3'] = None
                    else:
                         st.error("Fonction 'prepare_hist_data' non définie.")
                         st.session_state['export_fig_hist_compare_pm2_t3'] = None
                         st.session_state['export_fig_hist_compare_vf_t3'] = None

                    # --- Nuages de Points Comparatifs ---
                    st.markdown("---")
                    st.subheader("Analyse Détaillée 'À Vendre' (Surface vs Prix)")
                    fig_scatter_vf_anc = None; fig_scatter_pm2_anc = None # Init figures Ancien
                    fig_scatter_vf_neuf = None; fig_scatter_pm2_neuf = None # Init figures Neuf

                    # Section Ancien
                    st.markdown("#### Biens Anciens")
                    if not df_ancien_filtered_tab3.empty:
                        required_scatter_cols = ['surface_reelle_bati', 'valeur_fonciere', 'prix_m2', 'nombre_pieces_principales', 'nom_commune']
                        if all(col in df_ancien_filtered_tab3.columns for col in required_scatter_cols):
                            try:
                                df_ancien_scatter = df_ancien_filtered_tab3.copy()
                                if 'categorize_pieces' in locals() and callable(categorize_pieces):
                                     df_ancien_scatter['nombre_pieces_cat'] = df_ancien_scatter['nombre_pieces_principales'].apply(categorize_pieces)
                                     piece_order = ["=1", "=2", "=3", "=4", ">4", "Autre/Inconnu"]; valid_piece_cats_ancien = [cat for cat in piece_order if cat in df_ancien_scatter['nombre_pieces_cat'].unique()]
                                     col_ancien1, col_ancien2 = st.columns(2)
                                     with col_ancien1:
                                         st.write("**Surface vs Valeur Foncière**")
                                         fig_scatter_vf_anc = px.scatter(df_ancien_scatter, x="surface_reelle_bati", y="valeur_fonciere", color="nombre_pieces_cat", labels={"surface_reelle_bati": "Surface (m²)", "valeur_fonciere": "Valeur (€)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats_ancien}, hover_data=['nom_commune'], template="plotly_white") # Ajout template
                                         st.plotly_chart(fig_scatter_vf_anc, use_container_width=True, key='t3_scatter_vf_ancien')
                                     with col_ancien2:
                                         st.write("**Surface vs Prix m²**")
                                         fig_scatter_pm2_anc = px.scatter(df_ancien_scatter, x="surface_reelle_bati", y="prix_m2", color="nombre_pieces_cat", labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix m² (€/m²)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats_ancien}, hover_data=['nom_commune'], template="plotly_white") # Ajout template
                                         st.plotly_chart(fig_scatter_pm2_anc, use_container_width=True, key='t3_scatter_pm2_ancien')
                                else: st.error("Fonction 'categorize_pieces' non définie.")
                            except Exception as e: st.error(f"Err scatter Ancien: {e}"); logging.error(f"Err scatter Ancien T3: {e}", exc_info=True)
                        else: st.warning(f"Colonnes manquantes pour scatter Ancien: {', '.join([c for c in required_scatter_cols if c not in df_ancien_filtered_tab3.columns])}")
                    else: st.info("Aucune donnée 'Ancien' filtrée pour nuages.")
                    # STOCKAGE pour Export Zip
                    st.session_state['export_fig_scatter_vf_anc_t3'] = fig_scatter_vf_anc
                    st.session_state['export_fig_scatter_pm2_anc_t3'] = fig_scatter_pm2_anc

                    # Section Neuf
                    st.markdown("#### Biens Neufs")
                    if not df_neuf_filtered_tab3.empty:
                        required_scatter_cols = ['surface_reelle_bati', 'valeur_fonciere', 'prix_m2', 'nombre_pieces_principales', 'nom_commune']
                        if all(col in df_neuf_filtered_tab3.columns for col in required_scatter_cols):
                            try:
                                df_neuf_scatter = df_neuf_filtered_tab3.copy()
                                if 'categorize_pieces' in locals() and callable(categorize_pieces):
                                    df_neuf_scatter['nombre_pieces_cat'] = df_neuf_scatter['nombre_pieces_principales'].apply(categorize_pieces)
                                    piece_order = ["=1", "=2", "=3", "=4", ">4", "Autre/Inconnu"]; valid_piece_cats_neuf = [cat for cat in piece_order if cat in df_neuf_scatter['nombre_pieces_cat'].unique()]
                                    col_neuf1, col_neuf2 = st.columns(2)
                                    with col_neuf1:
                                        st.write("**Surface vs Valeur Foncière**")
                                        fig_scatter_vf_neuf = px.scatter(df_neuf_scatter, x="surface_reelle_bati", y="valeur_fonciere", color="nombre_pieces_cat", labels={"surface_reelle_bati": "Surface (m²)", "valeur_fonciere": "Valeur (€)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats_neuf}, hover_data=['nom_commune'], template="plotly_white") # Ajout template
                                        st.plotly_chart(fig_scatter_vf_neuf, use_container_width=True, key='t3_scatter_vf_neuf')
                                    with col_neuf2:
                                        st.write("**Surface vs Prix m²**")
                                        fig_scatter_pm2_neuf = px.scatter(df_neuf_scatter, x="surface_reelle_bati", y="prix_m2", color="nombre_pieces_cat", labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix m² (€/m²)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats_neuf}, hover_data=['nom_commune'], template="plotly_white") # Ajout template
                                        st.plotly_chart(fig_scatter_pm2_neuf, use_container_width=True, key='t3_scatter_pm2_neuf')
                                else: st.error("Fonction 'categorize_pieces' non définie.")
                            except Exception as e: st.error(f"Err scatter Neuf: {e}"); logging.error(f"Err scatter Neuf T3: {e}", exc_info=True)
                        else: st.warning(f"Colonnes manquantes pour scatter Neuf: {', '.join([c for c in required_scatter_cols if c not in df_neuf_filtered_tab3.columns])}")
                    else: st.info("Aucune donnée 'Neuf' filtrée pour nuages.")
                    # STOCKAGE pour Export Zip
                    st.session_state['export_fig_scatter_vf_neuf_t3'] = fig_scatter_vf_neuf
                    st.session_state['export_fig_scatter_pm2_neuf_t3'] = fig_scatter_pm2_neuf

                # --- Fin de la condition if has_dvf and (has_ancien or has_neuf): ---
                # else: # Le reset est fait au début de la condition

            # --- Fin de with tab3: ---


            # ======================================================
            # ========= ONGLET: Analyse Concurrence Achat ========= (Nouveau)
            # ======================================================
            # ======================================================
            # ========= ONGLET: Analyse Concurrence Achat ========= (Code Modifié et Complet v5)
            # ======================================================
            # ======================================================
            # ========= ONGLET: Analyse Concurrence Achat ========= (Code Modifié et Complet v7 - Revert Loading + Graphs)
            # ======================================================
            # ======================================================
            # ========= ONGLET: Analyse Concurrence Achat ========= (Code Modifié et Complet v8 - Colonnes pour Graphiques)
            # ======================================================
            # ======================================================
            # ========= ONGLET: Analyse Concurrence Achat ========= (Code Modifié et Complet v9 - Métriques par Type)
            # ======================================================
            with tab4:
                st.header("🛒 Analyse Concurrence Achat")
                st.write("Analysez quels types de biens sont disponibles sur le marché 'À Vendre' pour un budget donné dans une zone géographique sélectionnée.")

                # --- 1. Récupération des Données "À Vendre" (depuis état Tab 3) ---
                # (Code inchangé)
                df_ancien_sale_raw = st.session_state.get('df_ancien_tab3_raw', pd.DataFrame())
                df_neuf_sale_raw = st.session_state.get('df_neuf_tab3_raw', pd.DataFrame())

                if df_ancien_sale_raw.empty and df_neuf_sale_raw.empty:
                    st.warning("Veuillez charger au moins un fichier MoteurImmo ('Ancien' ou 'Neuf') dans l'onglet 'Comparaison Marché' pour utiliser cette analyse.")
                    # Reset SEULEMENT les clés des 3 barres si pas de données initiales
                    st.session_state['export_fig_apt_t4'] = None
                    st.session_state['export_fig_mdv_t4'] = None # Note: Clé basée sur variable fig_mdv
                    st.session_state['export_fig_mat_t4'] = None # Note: Clé basée sur variable fig_mat
                else:
                    # Combinaison et préparation des données (inchangé)
                    df_for_sale_combined = pd.concat([df_ancien_sale_raw, df_neuf_sale_raw], ignore_index=True).copy()
                    logging.info(f"Tab4: Données 'A Vendre' combinées (depuis état Tab 3) - {len(df_for_sale_combined)} lignes.")
                    num_cols_t4 = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain', 'prix_m2']
                    for col in num_cols_t4:
                        if col in df_for_sale_combined.columns: df_for_sale_combined[col] = pd.to_numeric(df_for_sale_combined[col], errors='coerce')

                    # --- 2. Classification "Maison de Village" ---
                    # (Code inchangé)
                    def classify_detailed_type(row):
                        type_loc = row.get('type_local', 'Inconnu'); surf_terr = pd.to_numeric(row.get('surface_terrain', 0), errors='coerce'); surf_terr = 0 if pd.isna(surf_terr) else surf_terr
                        if type_loc == 'Maison': return 'Maison de Village' if surf_terr <= 0 else 'Maison (avec terrain)'
                        elif type_loc == 'Appartement': return 'Appartement'
                        else: return str(type_loc)
                    if 'type_local' in df_for_sale_combined.columns:
                         if 'surface_terrain' not in df_for_sale_combined.columns: df_for_sale_combined['surface_terrain'] = 0
                         df_for_sale_combined['surface_terrain'] = pd.to_numeric(df_for_sale_combined['surface_terrain'], errors='coerce').fillna(0)
                         df_for_sale_combined['Type Détaillé'] = df_for_sale_combined.apply(classify_detailed_type, axis=1)
                         logging.info("Tab4: Classification 'Type Détaillé' appliquée.")
                    else: df_for_sale_combined['Type Détaillé'] = 'Type Local Manquant'; logging.warning("Tab4: Colonne 'type_local' manquante.")

                    # --- 3. Filtre Géographique Indépendant (Tab 4) ---
                    # (Code inchangé)
                    st.markdown("---"); st.subheader("A. Sélectionner la Zone Géographique")
                    communes_dispo_t4 = sorted(df_for_sale_combined['nom_commune'].unique()) if 'nom_commune' in df_for_sale_combined.columns else []

                    if not communes_dispo_t4:
                        st.warning("Aucune commune trouvée dans les données 'À Vendre'.")
                        df_sale_geo_filtered = pd.DataFrame()
                        selected_communes_t4 = [] # S'assurer que la variable existe même si pas de sélection
                    else:
                        key_communes_t4 = 't4_communes'
                        calculated_default_communes_t4 = communes_dispo_t4 # Default is all available

                        # --- Correction pour éviter l'avertissement ---
                        # Créer directement le widget. 'default' initialise l'état si key est nouvelle.
                        selected_communes_t4 = st.multiselect(
                            "Communes pour l'analyse :",
                            options=communes_dispo_t4,
                            default=calculated_default_communes_t4, # Le défaut pour la 1ère exécution
                            key=key_communes_t4 # Lie à st.session_state['t4_communes']
                        )
                        # --- Fin Correction ---

                        # --- Application du filtre (inchangé) ---
                        if selected_communes_t4 and 'nom_commune' in df_for_sale_combined.columns:
                            df_sale_geo_filtered = df_for_sale_combined[df_for_sale_combined['nom_commune'].isin(selected_communes_t4)].copy()
                        elif not selected_communes_t4:
                            st.info("Sélectionnez au moins une commune.")
                            df_sale_geo_filtered = pd.DataFrame()
                        else: # Sécurité
                            st.error("'nom_commune' manquante.")
                            df_sale_geo_filtered = pd.DataFrame()




                    # if not communes_dispo_t4:
                    #     st.warning("Aucune commune trouvée."); df_sale_geo_filtered = pd.DataFrame()
                    # else:
                    #     key_communes_t4 = 't4_communes'
                    #     if key_communes_t4 not in st.session_state: st.session_state[key_communes_t4] = communes_dispo_t4
                    #     validated_communes_t4 = [c for c in st.session_state.get(key_communes_t4, []) if c in communes_dispo_t4]
                    #     if not validated_communes_t4 and communes_dispo_t4: validated_communes_t4 = communes_dispo_t4
                    #     st.session_state[key_communes_t4] = validated_communes_t4
                    #     selected_communes_t4 = st.multiselect("Communes pour l'analyse :", communes_dispo_t4, default=st.session_state[key_communes_t4], key=key_communes_t4)
                    #     if selected_communes_t4 and 'nom_commune' in df_for_sale_combined.columns: df_sale_geo_filtered = df_for_sale_combined[df_for_sale_combined['nom_commune'].isin(selected_communes_t4)].copy()
                    #     elif not selected_communes_t4: st.info("Sélectionnez au moins une commune."); df_sale_geo_filtered = pd.DataFrame()
                    #     else: st.error("'nom_commune' manquante."); df_sale_geo_filtered = pd.DataFrame()


                    # --- NOUVELLE SECTION FILTRE PRIX (à insérer à la place de l'ancienne) ---
                    st.markdown("---")
                    st.subheader("B. Définir le Budget Cible")
                    df_to_filter_price = df_sale_geo_filtered # Utiliser les données déjà filtrées géographiquement

                    filter_key_price_t4 = 't4_filter_price' # Clé de base pour cet onglet
                    min_key_p_t4 = f"{filter_key_price_t4}_min"
                    max_key_p_t4 = f"{filter_key_price_t4}_max"
                    col_price = 'valeur_fonciere' # Colonne à filtrer
                    step_value = 10000 # Pas pour les inputs (ajustez si besoin)

                    # Initialiser les variables de sélection (important !)
                    selected_price_min_t4 = 0
                    selected_price_max_t4 = 1000000 # Mettre une valeur max haute par défaut

                    # Calculer les bornes min/max depuis les données disponibles
                    min_p_data, max_p_data = 0, 1000000 # Valeurs par défaut larges
                    if not df_to_filter_price.empty and col_price in df_to_filter_price.columns:
                        prix_valides = df_to_filter_price[col_price].dropna()
                        prix_valides = prix_valides[prix_valides > 0]
                        if not prix_valides.empty:
                            min_p_data = int(prix_valides.min())
                            max_p_data = int(prix_valides.max())
                            if min_p_data >= max_p_data: # S'assurer qu'il y a une plage
                                max_p_data = min_p_data + step_value

                    # Gérer les valeurs par défaut (persistance via session_state)
                    default_min_p = st.session_state.get(min_key_p_t4, min_p_data)
                    default_max_p = st.session_state.get(max_key_p_t4, max_p_data)
                    # S'assurer que les défauts sont dans les bornes des données actuelles
                    default_min_p = max(min_p_data, default_min_p)
                    default_max_p = min(max_p_data, default_max_p)
                    # S'assurer que max >= min
                    if default_max_p < default_min_p:
                        default_max_p = default_min_p

                    # Afficher les number inputs
                    st.write("Fourchette de Prix Total (€) :")
                    price_col_min, price_col_max = st.columns(2)
                    with price_col_min:
                        # Utiliser les variables initialisées plus haut pour value
                        selected_price_min_t4 = st.number_input(
                            "Min",
                            min_value=min_p_data,
                            max_value=max_p_data,
                            value=default_min_p, # La valeur par défaut calculée
                            step=step_value,
                            key=min_key_p_t4, # Utiliser la clé pour session state
                            format="%d"
                        )
                    with price_col_max:
                        # Le minimum pour le champ "Max" doit être la valeur sélectionnée pour "Min"
                        min_val_for_max = selected_price_min_t4 if selected_price_min_t4 is not None else min_p_data
                        # Utiliser les variables initialisées plus haut pour value
                        selected_price_max_t4 = st.number_input(
                            "Max",
                            min_value=min_val_for_max, # Minimum est la valeur min sélectionnée
                            max_value=max_p_data,
                            # Assurer que la valeur par défaut est au moins égale au min sélectionné
                            value=max(min_val_for_max, default_max_p),
                            step=step_value,
                            key=max_key_p_t4, # Utiliser la clé pour session state
                            format="%d"
                        )
                    # --- FIN NOUVELLE SECTION FILTRE PRIX ---


                    # # --- 4. Filtre par Prix Total (Slider) ---
                    # # (Code inchangé)
                    # st.markdown("---"); st.subheader("B. Définir le Budget Cible")
                    # df_to_filter_price = df_sale_geo_filtered; selected_target_price_range = (0,1)
                    # if not df_to_filter_price.empty and 'valeur_fonciere' in df_to_filter_price.columns:
                    #     prix_valides = df_to_filter_price['valeur_fonciere'].dropna(); prix_valides = prix_valides[prix_valides > 0]
                    #     if not prix_valides.empty:
                    #         min_p, max_p = int(prix_valides.min()), int(prix_valides.max()); min_slider_p, max_slider_p = min_p, max_p
                    #         if min_p == max_p: max_slider_p = max_p + 1000
                    #         calc_def_p = (min_p, max_p); key_price_range_t4 = 't4_price_range'
                    #         if key_price_range_t4 not in st.session_state: st.session_state[key_price_range_t4] = calc_def_p
                    #         current_value_p = st.session_state[key_price_range_t4]; bounded_value_p = (max(min_slider_p, current_value_p[0]), min(max_slider_p, current_value_p[1]))
                    #         selected_target_price_range = st.slider("Fourchette de Prix Total (€) :", min_value=min_slider_p, max_value=max_slider_p, value=bounded_value_p, key=key_price_range_t4, step=1000)
                    #     else: st.warning("Aucune valeur foncière valide."); selected_target_price_range = (0, 1)
                    # elif df_to_filter_price.empty: st.info("Aucune donnée pour les communes sélectionnées."); selected_target_price_range = (0, 1)
                    # else: st.error("'valeur_fonciere' manquante."); selected_target_price_range = (0, 1)

                    # --- 5. Application Filtre Prix et Ajout Tranche Surface ---
                    # (Code inchangé)
                    st.markdown("---"); st.subheader("C. Résultats : Biens Concurrents dans le Budget")
                    df_competition = pd.DataFrame()
                    if not df_sale_geo_filtered.empty and 'valeur_fonciere' in df_sale_geo_filtered.columns:
                        try:
                            # mask_price = (df_sale_geo_filtered['valeur_fonciere'] >= selected_target_price_range[0]) & (df_sale_geo_filtered['valeur_fonciere'] <= selected_target_price_range[1])
                            # df_competition = df_sale_geo_filtered[mask_price].copy()
                            mask_price = (df_sale_geo_filtered['valeur_fonciere'] >= selected_price_min_t4) & \
                            (df_sale_geo_filtered['valeur_fonciere'] <= selected_price_max_t4)

                         
                            df_competition = df_sale_geo_filtered[mask_price].copy()
                            logging.info(f"Tab4: {len(df_competition)} biens concurrents trouvés.")

                            if not df_competition.empty:
                                bins_apartments = [0, 20, 30, 50, 61, 81, float('inf')]; labels_apartments = ['<20 m²', '20-29 m²', '30-49 m²', '50-60 m²', '61-80 m²', '>80 m²']
                                bins_houses = [0, 40, 71, 101, float('inf')]; labels_houses = ['<40 m²', '40-70 m²', '71-100 m²', '>100 m²']
                                def assign_surface_bin(row):
                                    surf = pd.to_numeric(row.get('surface_reelle_bati', 0), errors='coerce'); surf = 0 if pd.isna(surf) else surf
                                    type_det = row.get('Type Détaillé', 'Inconnu')
                                    try:
                                        if type_det == 'Appartement': return pd.cut([surf], bins=bins_apartments, labels=labels_apartments, right=False, include_lowest=True)[0]
                                        elif type_det in ['Maison (avec terrain)', 'Maison de Village']: return pd.cut([surf], bins=bins_houses, labels=labels_houses, right=False, include_lowest=True)[0]
                                        else: return 'N/A'
                                    except IndexError: return "Hors Tranche"
                                    except Exception as e_cut: logging.error(f"Erreur pd.cut Tab4: {e_cut}"); return "Erreur Bin"
                                if 'surface_reelle_bati' in df_competition.columns and 'Type Détaillé' in df_competition.columns:
                                    df_competition['Tranche Surface'] = df_competition.apply(assign_surface_bin, axis=1)
                                    all_labels = sorted(list(set(labels_apartments) | set(labels_houses)) + ['N/A', 'Hors Tranche', 'Erreur Bin'])
                                    df_competition['Tranche Surface'] = pd.Categorical(df_competition['Tranche Surface'], categories=all_labels, ordered=True)
                                    logging.info("Tab4: 'Tranche Surface' ajoutée.")
                                else: df_competition['Tranche Surface'] = 'Info Manquante'; logging.warning("Tab4: Colonnes manquantes pour 'Tranche Surface'.")
                        except Exception as e: st.error(f"Erreur filtrage prix : {e}"); logging.error(f"Err filtrage prix T4: {e}", exc_info=True)

                    # --- 6. Affichage des Résultats ---
                    if not df_competition.empty:

                        # --- NOUVEAU : Métriques par Type ---
                        st.markdown("##### Répartition des Biens Concurrents")
                        col_met1, col_met2, col_met3, col_met4 = st.columns(4) # 4 colonnes pour Total + 3 Types

                        # Calculer les comptes par type détaillé
                        type_counts = df_competition['Type Détaillé'].value_counts() if 'Type Détaillé' in df_competition.columns else pd.Series(dtype=int)

                        with col_met1:
                            st.metric("Total Biens", len(df_competition))
                        with col_met2:
                            st.metric("Appartements", type_counts.get('Appartement', 0))
                        with col_met3:
                            st.metric("Maisons Village", type_counts.get('Maison de Village', 0))
                        with col_met4:
                            st.metric("Maisons Terrain", type_counts.get('Maison (avec terrain)', 0))
                        # --- FIN NOUVEAU ---

                        st.markdown("---") # Séparateur

                        # --- Tableau Récapitulatif ---
                        display_summary = False
                        df_summary = pd.DataFrame(columns=['Type Détaillé', 'Tranche Surface', 'Nombre de Biens'])
                        if 'Type Détaillé' in df_competition.columns and 'Tranche Surface' in df_competition.columns:
                            try:
                                df_summary = df_competition.groupby(['Type Détaillé', 'Tranche Surface'], observed=False).agg(Nombre_de_Biens=('valeur_fonciere', 'count')).reset_index()
                                df_summary = df_summary[df_summary['Nombre_de_Biens'] > 0].copy()
                                df_summary = df_summary.sort_values(by=['Type Détaillé', 'Tranche Surface'])
                                display_summary = not df_summary.empty
                            except Exception as e_group: st.error(f"Erreur tableau récapitulatif: {e_group}"); logging.error(f"Err groupby T4: {e_group}", exc_info=True)

                        if display_summary:
                            st.subheader("Répartition par Type et Tranche de Surface")
                            st.dataframe(df_summary, use_container_width=True, hide_index=True)

                            # --- STOCKAGE Tableau Récap pour Export Zip ---
                            # (Note: C'est le df_summary qui est affiché, pas df_competition_summary comme avant)
                            # Utilisons une clé cohérente, même si le contenu est un peu différent
                            st.session_state['export_df_competition_summary_t4'] = df_summary.copy()
                            logging.info(f"Tab 4: df_summary (répartition) stocké.")
                            # --- FIN STOCKAGE ---

                            # --- Graphiques à Barres par Type (en colonnes) ---
                            st.subheader("Visualisation de la Répartition par Surface")
                            col_graph1, col_graph2, col_graph3 = st.columns(3)

                            category_order_apt = labels_apartments
                            category_order_house = labels_houses

                            # Initialiser les figures à None
                            fig_apt = None; fig_mdv = None; fig_mat = None

                            with col_graph1:
                                df_summary_apt = df_summary[df_summary['Type Détaillé'] == 'Appartement']
                                if not df_summary_apt.empty:
                                    try: # (Code graphique Apt inchangé)
                                        fig_apt = px.bar(df_summary_apt, x='Tranche Surface', y='Nombre_de_Biens', title='Appartements', labels={'Nombre_de_Biens': 'Nb Biens'}, category_orders={'Tranche Surface': category_order_apt}, template="plotly_white")
                                        fig_apt.update_layout(xaxis_title=None, yaxis_title="Nb Biens", title_x=0.5)
                                        st.plotly_chart(fig_apt, use_container_width=True)
                                    except Exception as e: st.error(f"Err Graphe Apt: {e}"); logging.error(f"Err fig apt T4: {e}", exc_info=True)
                                else: st.caption("Aucun appartement.")
                                # --- STOCKAGE pour Export Zip ---
                                st.session_state['export_fig_apt_t4'] = fig_apt
                                # --- FIN STOCKAGE ---

                            with col_graph2:
                                df_summary_mdv = df_summary[df_summary['Type Détaillé'] == 'Maison de Village']
                                if not df_summary_mdv.empty:
                                    try: # (Code graphique MdV inchangé)
                                        fig_mdv = px.bar(df_summary_mdv, x='Tranche Surface', y='Nombre_de_Biens', title='Maisons de Village', labels={'Nombre_de_Biens': 'Nb Biens'}, category_orders={'Tranche Surface': category_order_house}, template="plotly_white")
                                        fig_mdv.update_layout(xaxis_title=None, yaxis_title=None, title_x=0.5)
                                        st.plotly_chart(fig_mdv, use_container_width=True)
                                    except Exception as e: st.error(f"Err Graphe MdV: {e}"); logging.error(f"Err fig mdv T4: {e}", exc_info=True)
                                else: st.caption("Aucune maison de village.")
                                # --- STOCKAGE pour Export Zip (Utilise clé 'mai' pour Maison) ---
                                st.session_state['export_fig_mdv_t4'] = fig_mdv # Stocke fig_mdv sous clé 'mai'
                                # --- FIN STOCKAGE ---


                            with col_graph3:
                                df_summary_mat = df_summary[df_summary['Type Détaillé'] == 'Maison (avec terrain)']
                                if not df_summary_mat.empty:
                                    try: # (Code graphique MaT inchangé)
                                        fig_mat = px.bar(df_summary_mat, x='Tranche Surface', y='Nombre_de_Biens', title='Maisons avec Terrain', labels={'Nombre_de_Biens': 'Nb Biens'}, category_orders={'Tranche Surface': category_order_house}, template="plotly_white")
                                        fig_mat.update_layout(xaxis_title=None, yaxis_title=None, title_x=0.5)
                                        st.plotly_chart(fig_mat, use_container_width=True)
                                    except Exception as e: st.error(f"Err Graphe MaT: {e}"); logging.error(f"Err fig mat T4: {e}", exc_info=True)
                                else: st.caption("Aucune maison avec terrain.")
                                # --- STOCKAGE pour Export Zip (Utilise clé 'terr' pour Terrain/MaT) ---
                                st.session_state['export_fig_mat_t4'] = fig_mat # Stocke fig_mat sous clé 'terr'
                                # --- FIN STOCKAGE ---


                        else:
                            st.caption("Impossible de générer le tableau récapitulatif (nécessaire pour les graphiques).")
                            # Reset des clés si le tableau n'est pas généré
                            st.session_state['export_df_competition_summary_t4'] = None
                            st.session_state['export_fig_apt_t4'] = None
                            st.session_state['export_fig_mdv_t4'] = None
                            st.session_state['export_fig_mat_t4'] = None


                        # --- Tableau Détaillé dans Expander ---
                        # (Code inchangé)
                        st.markdown("---")
                        with st.expander("Voir le détail des biens concurrents"):
                            cols_display_order = ['nom_commune', 'type_local', 'Type Détaillé', 'valeur_fonciere', 'prix_m2', 'surface_reelle_bati', 'Tranche Surface', 'nombre_pieces_principales', 'surface_terrain', 'Lien Annonce']
                            cols_to_show = [col for col in cols_display_order if col in df_competition.columns]
                            st.dataframe(
                                df_competition[cols_to_show],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Lien Annonce": st.column_config.LinkColumn( # <--- Assurez-vous que le nom ici correspond
                                        "Lien Annonce", # Label de la colonne dans Streamlit
                                        display_text="Ouvrir Lien", # Texte affiché pour le lien (ou None pour afficher l'URL)
                # Vous pouvez aussi régler la largeur max etc. ici si besoin
                # max_chars=100
                                    )
                                }
                            )

                            # --- Dans with tab4:, après l'expander ---

                            try:
                                # Assurer que df_competition et cols_to_show existent et sont valides
                                # (Vérifier si 'df_competition' et 'cols_to_show' sont définis et non vides/None)
                                df_exists = 'df_competition' in locals() and not df_competition.empty
                                cols_exist = 'cols_to_show' in locals() and cols_to_show

                                if df_exists and cols_exist:
                                    csv_competition = df_competition[cols_to_show].to_csv(index=False, encoding='utf-8-sig', sep=';')

                                    # --- MODIFICATION ICI: Utiliser les nouvelles variables min/max ---
                                    # Vérifier si les variables des number inputs existent avant de les utiliser
                                    min_price_for_filename = selected_price_min_t4 if 'selected_price_min_t4' in locals() else 'min'
                                    max_price_for_filename = selected_price_max_t4 if 'selected_price_max_t4' in locals() else 'max'

                                    st.download_button(
                                        label="Télécharger le détail",
                                        data=csv_competition,
                                        # Remplacer selected_target_price_range[0] et [1] par les nouvelles variables
                                        file_name=f"concurrence_budget_{min_price_for_filename}_{max_price_for_filename}.csv",
                                        mime="text/csv",
                                        key='download_tab4' # Garder la même clé
                                    )
                                    # --- FIN MODIFICATION ---
                                else:
                                    # Ne pas afficher le bouton si pas de données à télécharger
                                    st.caption("Aucun détail à télécharger pour les filtres actuels.")

                            except NameError as e_name:
                                 # Attraper spécifiquement si une variable *prévue* manque
                                st.error(f"Erreur préparation téléchargement (Variable Manquante): {e_name}. Vérifiez la définition des variables de filtre.")
                                logging.error(f"NameError download Tab4: {e_name}", exc_info=True)
                            except Exception as e_dl:
                                # Attraper autres erreurs potentielles (ex: to_csv)
                                st.error(f"Erreur préparation téléchargement : {e_dl}")
                                logging.error(f"Erreur download Tab4: {e_dl}", exc_info=True)




                    else:
                        st.info("Aucun bien concurrent trouvé pour cette sélection de communes et cette fourchette de prix.")
                        # Reset des clés session_state pour cet onglet
                        for key in keys_to_reset_t4: st.session_state[key] = None

            # --- Fin de la condition 'if df_ancien_sale_raw.empty and df_neuf_sale_raw.empty:' ---
            # --- Fin de l'onglet Analyse Concurrence Achat ---

            # ===========================================
            # ========= NOUVEL ONGLET: Données Impôts ========= (Code Corrigé v2)
            # ===========================================
            # ===========================================
            # ========= NOUVEL ONGLET: Données Impôts ========= (Code Corrigé v3)
            # ===========================================
            # ===========================================
            # ========= ONGLET: Données Impôts ========= (Intégration pdf_processor.py v1)
            # ===========================================
            # ======================================================
            # ========= ONGLET 5: Analyse Données Ventes (Impôts) ========= (Tableau Stats Harmonisé + Stockage)
            # ======================================================
            with tab5:
                st.header("📄 Analyse des Données Ventes (PDF Impôts)")

                # --- Uploader DANS l'onglet ---
                uploaded_file_impots_tab5 = st.file_uploader(
                    "Chargez ici votre fichier PDF des Impôts (PF) - Données Ventes",
                    type=["pdf"],
                    key='upload_impots_tab5', # Clé unique
                    disabled=not PDF_PROCESSOR_AVAILABLE # Désactiver si import a échoué
                )
                if not PDF_PROCESSOR_AVAILABLE:
                     st.error("Le module 'pdf_processor.py' est introuvable. Le chargement PDF est désactivé.")

                # --- Logique de traitement utilisant la fonction importée ---
                session_key_last_impots_pdf = 'last_processed_impots_pdf_name' # Clé pour suivre le fichier

                # Vérifier si un nouveau fichier est chargé ET si le processeur est dispo
                if uploaded_file_impots_tab5 is not None and PDF_PROCESSOR_AVAILABLE:
                    # Traiter seulement si le nom du fichier a changé
                    if st.session_state.get(session_key_last_impots_pdf) != uploaded_file_impots_tab5.name:
                        with st.spinner("Traitement du PDF Impôts en cours... (peut prendre du temps)"):
                            try:
                                # Assurer que la fonction est définie
                                if 'charger_et_traiter_pdf_ventes' in locals() and callable(charger_et_traiter_pdf_ventes):
                                    df_processed = charger_et_traiter_pdf_ventes(
                                        uploaded_file_impots_tab5,
                                        pages_a_lire='2-end' # Ajustez si nécessaire
                                    )
                                    if not df_processed.empty:
                                        st.session_state['df_impots'] = df_processed
                                        st.session_state['impots_data_loaded'] = True
                                        st.session_state[session_key_last_impots_pdf] = uploaded_file_impots_tab5.name # Mémoriser nom
                                        st.success(f"Données Impôts chargées et traitées ({len(df_processed)} lignes).")
                                        # st.rerun() # Décommenter si les filtres/graphes ne se MAJ pas automatiquement
                                    else:
                                        st.session_state['df_impots'] = pd.DataFrame()
                                        st.session_state['impots_data_loaded'] = False
                                        st.session_state[session_key_last_impots_pdf] = None # Oublier si échec/vide
                                        st.warning("Le traitement du PDF n'a retourné aucune donnée valide ou un DataFrame vide.")
                                else:
                                    st.error("Fonction 'charger_et_traiter_pdf_ventes' non définie.")
                                    st.session_state['df_impots'] = pd.DataFrame()
                                    st.session_state['impots_data_loaded'] = False
                                    st.session_state[session_key_last_impots_pdf] = None

                            except Exception as e_process:
                                st.error(f"Erreur lors du traitement du PDF Impôts : {e_process}")
                                logging.error(f"Erreur appel charger_et_traiter_pdf_ventes: {e_process}", exc_info=True)
                                st.session_state['df_impots'] = pd.DataFrame()
                                st.session_state['impots_data_loaded'] = False
                                st.session_state[session_key_last_impots_pdf] = None # Oublier si erreur

                # --- Vérifier l'état APRÈS tentative de chargement ---
                impots_data_available = st.session_state.get('impots_data_loaded', False)
                df_impots_loaded = st.session_state.get('df_impots', pd.DataFrame())

                # --- Initialiser/Reset clés session_state pour cet onglet ---
                keys_to_reset_t5 = [k for k in export_keys if k.endswith('_t5')] # Utilise la liste globale export_keys
                if not impots_data_available or df_impots_loaded.empty:
                    if uploaded_file_impots_tab5 is None and PDF_PROCESSOR_AVAILABLE:
                        st.info("⬆️ Veuillez charger un fichier PDF Impôts ci-dessus pour activer l'analyse de cet onglet.")
                    elif not PDF_PROCESSOR_AVAILABLE:
                        pass # Message d'erreur déjà affiché
                    else: # Fichier chargé mais vide ou erreur traitement
                        st.warning("Aucune donnée Impôts valide n'est actuellement chargée.")
                    for key in keys_to_reset_t5: st.session_state[key] = None # Reset si pas de données
                else:
                    # Les données sont là et non vides
                    df_tab_impots_base = df_impots_loaded.copy()

                    # --- Filtres spécifiques à cet onglet ---
                    st.markdown("---")
                    st.subheader("Filtres sur les Données Impôts")
                    col_f_impots1, col_f_impots2 = st.columns(2)
                    DEFAULT_RANGE_IMPOTS = (0, 1)

                    # Initialiser les variables de sélection
                    selected_annee_vente_impots = DEFAULT_RANGE_IMPOTS
                    selected_vf_impots_min, selected_vf_impots_max = 0, 10000000
                    selected_pm2_impots_min, selected_pm2_impots_max = 0, 50000
                    selected_surf_impots_min, selected_surf_impots_max = 0, 10000
                    selected_pieces_imp = DEFAULT_RANGE_IMPOTS
                    selected_communes_imp = []
                    selected_annee_constr_min, selected_annee_constr_max = 1900, datetime.datetime.now().year # Initialisation large

                    # --- Colonne Filtres 1 ---
                    with col_f_impots1:
                        # Slider Année de Vente
                        slider_key_annee_vente_imp = 't_impots_slider_annee_vente'
                        col_date_mut_imp = 'date_mutation'
                        current_year = datetime.datetime.now().year
                        fallback_min_year = current_year - 5; fallback_max_year = current_year + 1
                        if fallback_max_year <= fallback_min_year: fallback_max_year = fallback_min_year + 1
                        min_slider_an_vente_imp = fallback_min_year; max_slider_an_vente_imp = fallback_max_year
                        calculated_default_annee_vente_imp = (fallback_min_year, fallback_max_year)
                        if col_date_mut_imp in df_tab_impots_base.columns:
                            try:
                                annees_vente = pd.to_datetime(df_tab_impots_base[col_date_mut_imp], errors='coerce').dt.year.dropna().astype('Int64')
                                if not annees_vente.empty:
                                    min_calc = int(annees_vente.min()); max_calc = int(annees_vente.max())
                                    if 1900 <= min_calc <= 2100 and 1900 <= max_calc <= 2100:
                                        min_slider_an_vente_imp = min_calc; max_slider_an_vente_imp = max_calc
                                        if min_slider_an_vente_imp >= max_slider_an_vente_imp: max_slider_an_vente_imp = min_slider_an_vente_imp + 1
                                        calculated_default_annee_vente_imp = (min_slider_an_vente_imp, max_slider_an_vente_imp)
                                    else: logging.warning(f"Années vente Impots ({min_calc}-{max_calc}) hors limites.")
                                else: logging.info("Pas d'années de vente Impots valides.")
                            except Exception as e_parse_year: logging.error(f"Erreur calcul bornes année vente Impots: {e_parse_year}.")

                        # --- Correction pour éviter l'avertissement ---
                        # 1. Lire l'état actuel ou utiliser le défaut calculé
                        current_value_or_default_annee_vente = st.session_state.get(slider_key_annee_vente_imp, calculated_default_annee_vente_imp)

                        # 2. Borner la valeur
                        bounded_value_annee_vente_imp = (
                            max(min_slider_an_vente_imp, current_value_or_default_annee_vente[0]),
                            min(max_slider_an_vente_imp, current_value_or_default_annee_vente[1])
                        )
                        if bounded_value_annee_vente_imp[1] < bounded_value_annee_vente_imp[0]:
                            bounded_value_annee_vente_imp = (bounded_value_annee_vente_imp[0], bounded_value_annee_vente_imp[0])

                        # 3. Créer le slider
                        selected_annee_vente_impots = st.slider(
                            "Année de Vente:",
                            min_value=min_slider_an_vente_imp,
                            max_value=max_slider_an_vente_imp,
                            value=bounded_value_annee_vente_imp, # Utiliser la valeur bornée
                            key=slider_key_annee_vente_imp      # Lier à l'état
                        )
                        # --- Fin Correction ---

                        # if slider_key_annee_vente_imp not in st.session_state: st.session_state[slider_key_annee_vente_imp] = calculated_default_annee_vente_imp
                        # current_value_annee_vente_imp = st.session_state[slider_key_annee_vente_imp]
                        # bounded_value_annee_vente_imp = (max(min_slider_an_vente_imp, current_value_annee_vente_imp[0]), min(max_slider_an_vente_imp, current_value_annee_vente_imp[1]))
                        # if bounded_value_annee_vente_imp[1] < bounded_value_annee_vente_imp[0]: bounded_value_annee_vente_imp = (bounded_value_annee_vente_imp[0], bounded_value_annee_vente_imp[0])
                        # selected_annee_vente_impots = st.slider("Année de Vente:", min_value=min_slider_an_vente_imp, max_value=max_slider_an_vente_imp, value=bounded_value_annee_vente_imp, key=slider_key_annee_vente_imp)

                        st.markdown("---")

                        # Filtre VF Impôts (Number Inputs)
                        filter_key_vf_imp = 't5_filter_vf'; min_key_vf = f"{filter_key_vf_imp}_min"; max_key_vf = f"{filter_key_vf_imp}_max"
                        col_vf = 'valeur_fonciere'; min_vf_data, max_vf_data = 0, 10000000
                        if col_vf in df_tab_impots_base.columns and not df_tab_impots_base[col_vf].dropna().empty:
                            vf_valides = df_tab_impots_base[df_tab_impots_base[col_vf] > 0][col_vf].dropna()
                            if not vf_valides.empty: min_vf_data, max_vf_data = int(vf_valides.min()), int(vf_valides.max())
                        default_min_vf = st.session_state.get(min_key_vf, min_vf_data); default_max_vf = st.session_state.get(max_key_vf, max_vf_data)
                        default_min_vf = max(min_vf_data, default_min_vf); default_max_vf = min(max_vf_data, default_max_vf)
                        if default_max_vf < default_min_vf: default_max_vf = default_min_vf
                        st.write("Valeur Foncière (€) :"); vf_col_min_imp, vf_col_max_imp = st.columns(2)
                        with vf_col_min_imp: selected_vf_impots_min = st.number_input("Min VF Imp.", min_value=min_vf_data, max_value=max_vf_data, value=default_min_vf, step=10000, key=min_key_vf, format="%d", label_visibility="collapsed")
                        with vf_col_max_imp: min_val_for_max_vf = selected_vf_impots_min if selected_vf_impots_min is not None else min_vf_data; selected_vf_impots_max = st.number_input("Max VF Imp.", min_value=min_val_for_max_vf, max_value=max_vf_data, value=max(min_val_for_max_vf, default_max_vf), step=10000, key=max_key_vf, format="%d", label_visibility="collapsed")
                        st.markdown("---")

                        # Filtre Prix/m² Impôts (Number Inputs)
                        filter_key_pm2_imp = 't5_filter_pm2'; min_key_pm2 = f"{filter_key_pm2_imp}_min"; max_key_pm2 = f"{filter_key_pm2_imp}_max"
                        col_pm2 = 'prix_m2'; min_pm2_data, max_pm2_data = 0, 50000
                        if col_pm2 in df_tab_impots_base.columns and not df_tab_impots_base[col_pm2].dropna().empty:
                            pm2_valides = df_tab_impots_base[df_tab_impots_base[col_pm2] > 0][col_pm2].dropna()
                            if not pm2_valides.empty: min_pm2_data, max_pm2_data = int(pm2_valides.min()), int(pm2_valides.max())
                        default_min_pm2 = st.session_state.get(min_key_pm2, min_pm2_data); default_max_pm2 = st.session_state.get(max_key_pm2, max_pm2_data)
                        default_min_pm2 = max(min_pm2_data, default_min_pm2); default_max_pm2 = min(max_pm2_data, default_max_pm2)
                        if default_max_pm2 < default_min_pm2: default_max_pm2 = default_min_pm2
                        st.write("Prix m² (€/m²):"); pm2_col_min_imp, pm2_col_max_imp = st.columns(2)
                        with pm2_col_min_imp: selected_pm2_impots_min = st.number_input("Min PM2 Imp.", min_value=min_pm2_data, max_value=max_pm2_data, value=default_min_pm2, step=100, key=min_key_pm2, format="%d", label_visibility="collapsed")
                        with pm2_col_max_imp: min_val_for_max_pm2 = selected_pm2_impots_min if selected_pm2_impots_min is not None else min_pm2_data; selected_pm2_impots_max = st.number_input("Max PM2 Imp.", min_value=min_val_for_max_pm2, max_value=max_pm2_data, value=max(min_val_for_max_pm2, default_max_pm2), step=100, key=max_key_pm2, format="%d", label_visibility="collapsed")

                    # --- Colonne Filtres 2 ---
                    with col_f_impots2:
                        # Filtre Surface Réelle Bâti Impôts (Number Inputs)
                        filter_key_surf_imp = 't5_filter_surf'; min_key_surf = f"{filter_key_surf_imp}_min"; max_key_surf = f"{filter_key_surf_imp}_max"
                        col_surf = 'surface_reelle_bati'; min_surf_data, max_surf_data = 0, 10000
                        if col_surf in df_tab_impots_base.columns and not df_tab_impots_base[col_surf].dropna().empty:
                            surf_valides = df_tab_impots_base[df_tab_impots_base[col_surf] > 0][col_surf].dropna()
                            if not surf_valides.empty: min_surf_data, max_surf_data = int(surf_valides.min()), int(surf_valides.max())
                        default_min_surf = st.session_state.get(min_key_surf, min_surf_data); default_max_surf = st.session_state.get(max_key_surf, max_surf_data)
                        default_min_surf = max(min_surf_data, default_min_surf); default_max_surf = min(max_surf_data, default_max_surf)
                        if default_max_surf < default_min_surf: default_max_surf = default_min_surf
                        st.write("Surface Réelle Bâti (m²):"); surf_col_min_imp, surf_col_max_imp = st.columns(2)
                        with surf_col_min_imp: selected_surf_impots_min = st.number_input("Min Surf Imp.", min_value=min_surf_data, max_value=max_surf_data, value=default_min_surf, step=5, key=min_key_surf, format="%d", label_visibility="collapsed")
                        with surf_col_max_imp: min_val_for_max_surf = selected_surf_impots_min if selected_surf_impots_min is not None else min_surf_data; selected_surf_impots_max = st.number_input("Max Surf Imp.", min_value=min_val_for_max_surf, max_value=max_surf_data, value=max(min_val_for_max_surf, default_max_surf), step=5, key=max_key_surf, format="%d", label_visibility="collapsed")
                        st.markdown("---")

                        # Filtre Nombre de Pièces Impôts (Slider)
                        slider_key_pieces_imp = 't_impots_slider_pieces'
                        col_pieces_imp = 'nombre_pieces_principales'
                        min_slider_pieces_imp, max_slider_pieces_imp = DEFAULT_RANGE_IMPOTS[0], DEFAULT_RANGE_IMPOTS[1]+1
                        calculated_default_pieces_imp = DEFAULT_RANGE_IMPOTS
                        if col_pieces_imp in df_tab_impots_base.columns and df_tab_impots_base[col_pieces_imp].notna().any():
                            pieces_imp_num = pd.to_numeric(df_tab_impots_base[col_pieces_imp], errors='coerce').dropna()
                            pieces_imp_num = pieces_imp_num[pieces_imp_num > 0]
                            if not pieces_imp_num.empty:
                                min_pieces_data_imp = int(pieces_imp_num.min()); max_pieces_data_imp = int(pieces_imp_num.max())
                                min_slider_pieces_imp = min_pieces_data_imp; max_slider_pieces_imp = max_pieces_data_imp
                                if min_pieces_data_imp == max_pieces_data_imp: max_slider_pieces_imp += 1
                                calculated_default_pieces_imp = (min_pieces_data_imp, max_pieces_data_imp)
                            else: min_slider_pieces_imp, max_slider_pieces_imp = DEFAULT_RANGE_IMPOTS[0], DEFAULT_RANGE_IMPOTS[1]+1; calculated_default_pieces_imp = DEFAULT_RANGE_IMPOTS
                        else: min_slider_pieces_imp, max_slider_pieces_imp = DEFAULT_RANGE_IMPOTS[0], DEFAULT_RANGE_IMPOTS[1]+1; calculated_default_pieces_imp = DEFAULT_RANGE_IMPOTS


                        # --- Correction pour éviter l'avertissement ---
                        # 1. Lire l'état actuel ou utiliser le défaut calculé
                        current_value_or_default_pieces = st.session_state.get(slider_key_pieces_imp, calculated_default_pieces_imp)

                        # 2. Borner la valeur
                        bounded_value_pieces_imp = (
                            max(min_slider_pieces_imp, current_value_or_default_pieces[0]),
                            min(max_slider_pieces_imp, current_value_or_default_pieces[1])
                        )
                        if bounded_value_pieces_imp[1] < bounded_value_pieces_imp[0]:
                            bounded_value_pieces_imp = (bounded_value_pieces_imp[0], bounded_value_pieces_imp[0])

                        # 3. Créer le slider
                        selected_pieces_imp = st.slider(
                            "Nombre de Pièces :",
                            min_value=min_slider_pieces_imp,
                            max_value=max_slider_pieces_imp,
                            value=bounded_value_pieces_imp, # Utiliser la valeur bornée
                            key=slider_key_pieces_imp      # Lier à l'état
                        )
                        # --- Fin Correction ---

                        # if slider_key_pieces_imp not in st.session_state: st.session_state[slider_key_pieces_imp] = calculated_default_pieces_imp
                        # current_value_pieces_imp = st.session_state[slider_key_pieces_imp]
                        # bounded_value_pieces_imp = (max(min_slider_pieces_imp, current_value_pieces_imp[0]), min(max_slider_pieces_imp, current_value_pieces_imp[1]))
                        # if bounded_value_pieces_imp[1] < bounded_value_pieces_imp[0]: bounded_value_pieces_imp = (bounded_value_pieces_imp[0], bounded_value_pieces_imp[0])
                        # selected_pieces_imp = st.slider("Nombre de Pièces :", min_value=min_slider_pieces_imp, max_value=max_slider_pieces_imp, value=bounded_value_pieces_imp, key=slider_key_pieces_imp)
                       
                        st.markdown("---")

                        # Filtre Communes (Impôts)
                        multi_key_commune_imp = 't_impots_multi_commune'
                        col_commune_imp = 'nom_commune'
                        communes_imp = []
                        if col_commune_imp in df_tab_impots_base.columns:
                            communes_imp = sorted(df_tab_impots_base[col_commune_imp].dropna().unique())
                        if communes_imp:
                            calculated_default_commune_imp = communes_imp

                            # --- Correction pour éviter l'avertissement ---
                            # Créer directement le widget. 'default' initialise l'état si key est nouvelle.
                            selected_communes_imp = st.multiselect(
                                "Communes (Impôts) :",
                                options=communes_imp,
                                default=calculated_default_commune_imp, # Le défaut pour la 1ère exécution
                                key=multi_key_commune_imp # Lie à st.session_state['t_impots_multi_commune']
                            )
                            # --- Fin Correction ---

                            # if multi_key_commune_imp not in st.session_state: st.session_state[multi_key_commune_imp] = calculated_default_commune_imp
                            # current_selection_commune_imp = st.session_state.get(multi_key_commune_imp, [])
                            # validated_selection_commune_imp = [item for item in current_selection_commune_imp if item in communes_imp]
                            # if not validated_selection_commune_imp and communes_imp: validated_selection_commune_imp = communes_imp
                            # if validated_selection_commune_imp != current_selection_commune_imp: st.session_state[multi_key_commune_imp] = validated_selection_commune_imp
                            # selected_communes_imp = st.multiselect("Communes (Impôts) :", options=communes_imp, default=st.session_state[multi_key_commune_imp], key=multi_key_commune_imp)


                        else:
                            selected_communes_imp = []; st.info("Pas de communes trouvées (Impôts).")
                            if multi_key_commune_imp in st.session_state:
                                st.session_state[multi_key_commune_imp] = []
                            st.info("Pas de communes trouvées (Impôts).")


                        # <<< AJOUT DU FILTRE ANNÉE CONSTRUCTION >>>
                        st.markdown("---") # Séparateur
                        filter_key_annee_constr = 't5_filter_annee_constr'
                        min_key_ac = f"{filter_key_annee_constr}_min"
                        max_key_ac = f"{filter_key_annee_constr}_max"
                        col_ac = 'annee_construction' # Nom de colonne après traitement PDF

                        # Calculer Min/Max depuis les données chargées
                        min_ac_data, max_ac_data = 1900, datetime.datetime.now().year # Défauts larges
                        if col_ac in df_tab_impots_base.columns and df_tab_impots_base[col_ac].notna().any():
                            ac_valides = pd.to_numeric(df_tab_impots_base[col_ac], errors='coerce').dropna()
                            # Filtrer années aberrantes (ex: > année actuelle + 1 ou < 1800)
                            current_year = datetime.datetime.now().year
                            ac_valides = ac_valides[(ac_valides >= 1800) & (ac_valides <= current_year + 1)]
                            if not ac_valides.empty:
                                min_ac_data = int(ac_valides.min())
                                max_ac_data = int(ac_valides.max())
                                if min_ac_data >= max_ac_data: # Assurer une plage
                                    max_ac_data = min_ac_data + 1

                        # Gérer état session et valeurs par défaut
                        default_min_ac = st.session_state.get(min_key_ac, min_ac_data)
                        default_max_ac = st.session_state.get(max_key_ac, max_ac_data)
                        # Borner les défauts par rapport aux données actuelles
                        default_min_ac = max(min_ac_data, default_min_ac)
                        default_max_ac = min(max_ac_data, default_max_ac)
                        if default_max_ac < default_min_ac: default_max_ac = default_min_ac

                        # Afficher les widgets number_input
                        st.write("Année de Construction :")
                        ac_col_min, ac_col_max = st.columns(2)
                        with ac_col_min:
                            selected_annee_constr_min = st.number_input(
                                "Min Année Constr.",
                                min_value=min_ac_data,
                                max_value=max_ac_data,
                                value=default_min_ac, # Utiliser défaut calculé/borné
                                step=1,
                                key=min_key_ac, # Clé pour session state
                                format="%d",
                                label_visibility="collapsed"
                            )
                        with ac_col_max:
                            # Le minimum pour le champ Max = valeur Min sélectionnée
                            min_val_for_max_ac = selected_annee_constr_min if selected_annee_constr_min is not None else min_ac_data
                            selected_annee_constr_max = st.number_input(
                                "Max Année Constr.",
                                min_value=min_val_for_max_ac,
                                max_value=max_ac_data,
                                # Assurer que valeur défaut >= min sélectionné
                                value=max(min_val_for_max_ac, default_max_ac),
                                step=1,
                                key=max_key_ac, # Clé pour session state
                                format="%d",
                                label_visibility="collapsed"
                            )
                        # <<< FIN AJOUT FILTRE ANNÉE CONSTRUCTION >>>




                    # --- Appliquer les filtres pour l'onglet Impôts ---
                    filtered_df_impots = pd.DataFrame() # Init
                    try:
                        mask_impots = pd.Series(True, index=df_tab_impots_base.index)
                        col_date_mut_imp = 'date_mutation'
                        if col_date_mut_imp in df_tab_impots_base.columns:
                            annee_vente_col = pd.to_datetime(df_tab_impots_base[col_date_mut_imp], errors='coerce').dt.year
                            mask_impots &= (annee_vente_col >= selected_annee_vente_impots[0]) & (annee_vente_col <= selected_annee_vente_impots[1]) & (annee_vente_col.notna())
                        if 'valeur_fonciere' in df_tab_impots_base.columns: mask_impots &= (df_tab_impots_base['valeur_fonciere'] >= selected_vf_impots_min) & (df_tab_impots_base['valeur_fonciere'] <= selected_vf_impots_max)
                        if 'prix_m2' in df_tab_impots_base.columns: mask_impots &= (df_tab_impots_base['prix_m2'].fillna(0) >= selected_pm2_impots_min) & (df_tab_impots_base['prix_m2'].fillna(0) <= selected_pm2_impots_max)
                        if 'surface_reelle_bati' in df_tab_impots_base.columns: mask_impots &= (df_tab_impots_base['surface_reelle_bati'].fillna(0) >= selected_surf_impots_min) & (df_tab_impots_base['surface_reelle_bati'].fillna(0) <= selected_surf_impots_max)
                        if 'nombre_pieces_principales' in df_tab_impots_base.columns: mask_impots &= (df_tab_impots_base['nombre_pieces_principales'] >= selected_pieces_imp[0]) & (df_tab_impots_base['nombre_pieces_principales'] <= selected_pieces_imp[1])
                        if selected_communes_imp and 'nom_commune' in df_tab_impots_base.columns: mask_impots &= (df_tab_impots_base['nom_commune'].isin(selected_communes_imp))
                        # <<< AJOUT : Appliquer le filtre Année Construction >>>
                        if 'annee_construction' in df_tab_impots_base.columns:
                            # Assurer que la colonne est numérique avant de filtrer
                            annee_constr_num = pd.to_numeric(df_tab_impots_base['annee_construction'], errors='coerce')
                            # Récupérer les valeurs des number_input (elles sont déjà définies plus haut)
                            mask_impots &= (annee_constr_num >= selected_annee_constr_min) & \
                                           (annee_constr_num <= selected_annee_constr_max) & \
                                           (annee_constr_num.notna())
                        # <<< FIN AJOUT >>>


                        filtered_df_impots = df_tab_impots_base[mask_impots].copy()
                        logging.info(f"Tab Impots: Après filtres, shape = {filtered_df_impots.shape}")
                        st.session_state['filtered_df_tab5'] = filtered_df_impots.copy() # Stocker DF filtré

                    except Exception as e_filter_impots:
                        st.error(f"Erreur lors de l'application des filtres Impôts: {e_filter_impots}")
                        logging.error(f"Erreur filtrage Tab Impots: {e_filter_impots}", exc_info=True)
                        filtered_df_impots = pd.DataFrame()
                        st.session_state['filtered_df_tab5'] = pd.DataFrame()

                    # --- Affichages et Graphiques pour l'onglet Impôts ---
                    st.markdown("---")
                    nombre_biens_impots = len(filtered_df_impots)

                    if nombre_biens_impots > 0:

                        #   # --- AJOUT DEBUG ---
                        # st.write("DEBUG Tab 5 - Colonnes DANS filtered_df_impots AVANT stats:", filtered_df_impots.columns)
                        # st.dataframe(filtered_df_impots.head()) # Afficher les premières lignes
                        # # --- FIN AJOUT DEBUG ---
                        # --- NOUVEAU BLOC STATS (Style Tab 1) ---
                        st.subheader("Statistiques Clés (Impôts)")
                        stats_calc_ok_t5 = False
                        df_stats_combined_for_export_t5 = pd.DataFrame() # Init pour export
                        stats_impots_dict = {} # Init dict pour affichage

                        try:
                            stats_impots_dict['Nb Ventes'] = nombre_biens_impots # Utiliser le nom spécifique
                            metrics_list_for_export_t5 = []
                            metrics_list_for_export_t5.append({'Métrique': 'Nombre de Ventes', 'Valeur': f"{nombre_biens_impots:,}".replace(',', ' ')})

                            vf_series_t5 = filtered_df_impots['valeur_fonciere'].dropna().pipe(lambda s: s[s > 0])
                            pm2_series_t5 = filtered_df_impots['prix_m2'].dropna().pipe(lambda s: s[s > 0])

                            # Fonction de formatage (peut être définie globalement)
                            def format_stat_t5(value):
                                 if pd.notna(value) and np.isfinite(value):
                                     try: return f"{int(round(value)):,}".replace(',', ' ')
                                     except: return str(value)
                                 return 'N/A'

                            # Remplir stats_impots_dict avec les valeurs FORMATTÉES
                            stats_impots_dict['Moyenne VF (€)'] = format_stat_t5(vf_series_t5.mean()) if not vf_series_t5.empty else 'N/A'
                            stats_impots_dict['Médiane VF (€)'] = format_stat_t5(vf_series_t5.median()) if not vf_series_t5.empty else 'N/A'
                            stats_impots_dict['Min VF (€)'] = format_stat_t5(vf_series_t5.min()) if not vf_series_t5.empty else 'N/A'
                            stats_impots_dict['Max VF (€)'] = format_stat_t5(vf_series_t5.max()) if not vf_series_t5.empty else 'N/A'
                            try: stats_impots_dict['P90 VF (€)'] = format_stat_t5(vf_series_t5.quantile(0.9)) if not vf_series_t5.empty and len(vf_series_t5) >= 10 else 'N/A'
                            except: stats_impots_dict['P90 VF (€)'] = 'N/A'

                            stats_impots_dict['Moyenne m² (€/m²)'] = format_stat_t5(pm2_series_t5.mean()) if not pm2_series_t5.empty else 'N/A'
                            stats_impots_dict['Médiane m² (€/m²)'] = format_stat_t5(pm2_series_t5.median()) if not pm2_series_t5.empty else 'N/A'
                            stats_impots_dict['Min m² (€/m²)'] = format_stat_t5(pm2_series_t5.min()) if not pm2_series_t5.empty else 'N/A'
                            stats_impots_dict['Max m² (€/m²)'] = format_stat_t5(pm2_series_t5.max()) if not pm2_series_t5.empty else 'N/A'
                            try: stats_impots_dict['P90 m² (€/m²)'] = format_stat_t5(pm2_series_t5.quantile(0.9)) if not pm2_series_t5.empty and len(pm2_series_t5) >= 10 else 'N/A'
                            except: stats_impots_dict['P90 m² (€/m²)'] = 'N/A'

                            # Construire DF pour export
                            export_order_t5 = [
                                'Nombre de Ventes', 'Moyenne VF (€)', 'Moyenne m² (€/m²)',
                                'Médiane VF (€)', 'Médiane m² (€/m²)', 'Min VF (€)', 'Min m² (€/m²)',
                                'Max VF (€)', 'Max m² (€/m²)', 'P90 VF (€)', 'P90 m² (€/m²)'
                            ]
                            for metric_name in export_order_t5:
                                if metric_name != 'Nombre de Ventes': # Déjà ajouté
                                    metrics_list_for_export_t5.append({'Métrique': metric_name, 'Valeur': stats_impots_dict.get(metric_name, 'N/A')})

                            df_stats_combined_for_export_t5 = pd.DataFrame(metrics_list_for_export_t5)
                            try: # Réordonner
                                df_stats_combined_for_export_t5['Métrique'] = pd.Categorical(df_stats_combined_for_export_t5['Métrique'], categories=export_order_t5, ordered=True)
                                df_stats_combined_for_export_t5 = df_stats_combined_for_export_t5.sort_values('Métrique').reset_index(drop=True)
                            except Exception as e_reorder: logging.warning(f"Impossible réordonner DF stats T5: {e_reorder}")
                            df_stats_combined_for_export_t5['Valeur'] = df_stats_combined_for_export_t5['Valeur'].fillna('N/A')
                            stats_calc_ok_t5 = True
                            logging.info("Tab 5: df_stats_combined_for_export_t5 créé.")

                        except Exception as e_stats_impots:
                            st.error(f"Erreur lors de la création des statistiques (Impôts): {e_stats_impots}")
                            logging.error(f"Erreur stats Tab Impots: {e_stats_impots}", exc_info=True)
                            stats_calc_ok_t5 = False

                        # --- STOCKAGE du DataFrame Combiné pour l'export ---
                        if stats_calc_ok_t5:
                            st.session_state['export_df_stats_detail_t5'] = df_stats_combined_for_export_t5.copy()
                            logging.info("DataFrame stats combiné (Tab 5) stocké.")
                        else:
                             st.session_state['export_df_stats_detail_t5'] = None
                        # --- FIN STOCKAGE Stats ---

                        # --- Affichage Streamlit (Style Tab 1) ---
                        if stats_calc_ok_t5:
                            imp_mcol1, imp_mcol2, imp_mcol3 = st.columns(3)
                            imp_mcol1.metric("Nombre de Ventes", stats_impots_dict.get('Nb Ventes', 'N/A'))
                            imp_mcol2.metric("Moyenne VF (€)", stats_impots_dict.get('Moyenne VF (€)', 'N/A'))
                            imp_mcol3.metric("Moyenne m² (€/m²)", stats_impots_dict.get('Moyenne m² (€/m²)', 'N/A'))

                            with st.expander("Voir les statistiques détaillées (Impôts)"):
                                # Recréer les DFs juste pour affichage
                                stats_detail_1_t5 = {k: stats_impots_dict.get(k, 'N/A') for k in ['Min VF (€)', 'Min m² (€/m²)']}
                                stats_detail_2_t5 = {k: stats_impots_dict.get(k, 'N/A') for k in ['Max VF (€)', 'Max m² (€/m²)']}
                                stats_detail_3_t5 = {k: stats_impots_dict.get(k, 'N/A') for k in ['P90 VF (€)', 'P90 m² (€/m²)', 'Médiane VF (€)', 'Médiane m² (€/m²)']}

                                df_disp1_t5 = pd.DataFrame(list(stats_detail_1_t5.items()), columns=['Statistique', 'Valeur'])
                                df_disp2_t5 = pd.DataFrame(list(stats_detail_2_t5.items()), columns=['Statistique', 'Valeur'])
                                df_disp3_t5 = pd.DataFrame(list(stats_detail_3_t5.items()), columns=['Statistique', 'Valeur'])

                                # Appliquer style et afficher
                                try:
                                    # Assurez-vous que la fonction apply_row_styles est définie globalement
                                    colors_min = [('#93c47d', '#FFFFFF', '#d9ead3', '#000000'), ('#93c47d', '#FFFFFF', '#d9ead3', '#000000')]
                                    colors_max = [('#ff0000', '#FFFFFF', '#f4cccc', '#000000'), ('#ff0000', '#FFFFFF', '#f4cccc', '#000000')]
                                    colors_p90_med = [ # Ordre: P90 VF, P90 PM2, Med VF, Med PM2
                                        ('#f9cb9c', '#000000', '#ffe5d9', '#000000'), ('#f9cb9c', '#000000', '#ffe5d9', '#000000'),
                                        ('#FFBF00', '#000000', '#FFEBCC', '#000000'), ('#FFBF00', '#000000', '#FFEBCC', '#000000')
                                    ]

                                    # Vérifier si DFs ne sont pas vides avant de styler/afficher
                                    scol1_t5, scol2_t5, scol3_t5 = st.columns(3)
                                    with scol1_t5:
                                        if not df_disp1_t5.empty:
                                            styled_df1_t5 = df_disp1_t5.style.apply(apply_row_styles, colors_list=colors_min, axis=None)
                                            st.dataframe(styled_df1_t5, use_container_width=True, hide_index=True)
                                        else: st.caption("-")
                                    with scol2_t5:
                                        if not df_disp2_t5.empty:
                                            styled_df2_t5 = df_disp2_t5.style.apply(apply_row_styles, colors_list=colors_max, axis=None)
                                            st.dataframe(styled_df2_t5, use_container_width=True, hide_index=True)
                                        else: st.caption("-")
                                    with scol3_t5:
                                        if not df_disp3_t5.empty:
                                            # S'assurer que df_disp3_t5 a le bon ordre pour les couleurs
                                            df_disp3_t5['Statistique'] = pd.Categorical(df_disp3_t5['Statistique'], categories=['P90 VF (€)', 'P90 m² (€/m²)', 'Médiane VF (€)', 'Médiane m² (€/m²)'], ordered=True)
                                            df_disp3_t5 = df_disp3_t5.sort_values('Statistique')
                                            styled_df3_t5 = df_disp3_t5.style.apply(apply_row_styles, colors_list=colors_p90_med, axis=None)
                                            st.dataframe(styled_df3_t5, use_container_width=True, hide_index=True)
                                        else: st.caption("-")

                                except NameError:
                                    st.error("Fonction 'apply_row_styles' non définie.")
                                    # Fallback affichage brut
                                    scol1_t5, scol2_t5, scol3_t5 = st.columns(3)
                                    with scol1_t5: st.dataframe(df_disp1_t5, hide_index=True, use_container_width=True)
                                    with scol2_t5: st.dataframe(df_disp2_t5, hide_index=True, use_container_width=True)
                                    with scol3_t5: st.dataframe(df_disp3_t5, hide_index=True, use_container_width=True)
                                except Exception as e_style_disp_t5:
                                    logging.error(f"Erreur style/affichage tables expander T5: {e_style_disp_t5}")
                                    st.warning("Erreur affichage tableaux détaillés stylisés (Impôts).")
                                    # Fallback affichage brut
                                    scol1_t5, scol2_t5, scol3_t5 = st.columns(3)
                                    with scol1_t5: st.dataframe(df_disp1_t5, hide_index=True, use_container_width=True)
                                    with scol2_t5: st.dataframe(df_disp2_t5, hide_index=True, use_container_width=True)
                                    with scol3_t5: st.dataframe(df_disp3_t5, hide_index=True, use_container_width=True)
                        # --- FIN NOUVEAU BLOC STATS ---

                        # --- Graphiques (Structure de temp2.txt, ajout template et stockage) ---
                        st.subheader("Évolution Annuelle Moyenne (par Année de Vente)")
                        fig_vf_yearly_imp = None; fig_pm2_yearly_imp = None # Init
                        try:
                            col_date_vente = 'date_mutation'; new_col_annee_vente = 'annee_vente'
                            if col_date_vente in filtered_df_impots.columns:
                                df_graph_yearly = filtered_df_impots.copy()
                                df_graph_yearly[col_date_vente] = pd.to_datetime(df_graph_yearly[col_date_vente], errors='coerce')
                                df_graph_yearly[new_col_annee_vente] = df_graph_yearly[col_date_vente].dt.year
                                yearly_stats_impots = df_graph_yearly.dropna(subset=[new_col_annee_vente])
                                if not yearly_stats_impots.empty:
                                    yearly_stats_impots[new_col_annee_vente] = yearly_stats_impots[new_col_annee_vente].astype(int)
                                    yearly_stats_impots_agg = yearly_stats_impots.groupby(new_col_annee_vente).agg({'valeur_fonciere': 'mean', 'prix_m2': 'mean'}).reset_index()
                                    yearly_stats_impots_agg[new_col_annee_vente] = yearly_stats_impots_agg[new_col_annee_vente].astype(str)
                                    imp_col_evo1, imp_col_evo2 = st.columns(2)
                                    with imp_col_evo1:
                                        fig_vf_yearly_imp = px.bar(yearly_stats_impots_agg, x=new_col_annee_vente, y='valeur_fonciere', title="VF Moyenne / An", labels={'valeur_fonciere': 'Valeur Moyenne (€)', new_col_annee_vente: 'Année de Vente'}, color_discrete_sequence=['#1f77b4'], template="plotly_white") # Ajout template
                                        st.plotly_chart(fig_vf_yearly_imp, use_container_width=True, key='t_impots_bar_vf_yearly')
                                    with imp_col_evo2:
                                        fig_pm2_yearly_imp = px.bar(yearly_stats_impots_agg, x=new_col_annee_vente, y='prix_m2', title="Prix m² Moyen / An", labels={'prix_m2': 'Prix m² Moyen (€/m²)', new_col_annee_vente: 'Année de Vente'}, color_discrete_sequence=['#ff7f0e'], template="plotly_white") # Ajout template
                                        st.plotly_chart(fig_pm2_yearly_imp, use_container_width=True, key='t_impots_bar_pm2_yearly')
                                else: st.info("Pas de données annuelles valides pour l'évolution par année de vente (Impôts).")
                            else: st.warning(f"Colonne '{col_date_vente}' non trouvée pour calculer l'année de vente (Impôts).")
                        except Exception as e_evo_impots: st.error(f"Erreur lors de la création des graphiques annuels (Impôts): {e_evo_impots}"); logging.error(f"Err graphes annuels vente Impots: {e_evo_impots}", exc_info=True)
                        # STOCKAGE pour Export Zip (Même si None)
                        st.session_state['export_fig_vf_yearly_imp_t5'] = fig_vf_yearly_imp
                        st.session_state['export_fig_pm2_yearly_imp_t5'] = fig_pm2_yearly_imp

                        st.subheader("Distribution des Biens (Impôts)")
                        fig_hist_vf_imp = None; fig_hist_pm2_imp = None # Init
                        try:
                            imp_col_hist1, imp_col_hist2 = st.columns(2)
                            with imp_col_hist1:
                                fig_hist_vf_imp = px.histogram(filtered_df_impots, x="valeur_fonciere", title="Par Valeur Foncière", labels={"valeur_fonciere": "Valeur Foncière (€)"}, marginal="box", opacity=0.7, nbins=50, template="plotly_white") # Ajout template
                                fig_hist_vf_imp.update_layout(yaxis_title="Nombre de Ventes", bargap=0.1)
                                st.plotly_chart(fig_hist_vf_imp, use_container_width=True, key='t_impots_hist_vf')
                            with imp_col_hist2:
                                fig_hist_pm2_imp = px.histogram(filtered_df_impots, x="prix_m2", title="Par Prix au m²", labels={"prix_m2": "Prix au m² (€/m²)"}, marginal="box", opacity=0.7, nbins=50, template="plotly_white") # Ajout template
                                fig_hist_pm2_imp.update_layout(yaxis_title="Nombre de Ventes", bargap=0.1)
                                st.plotly_chart(fig_hist_pm2_imp, use_container_width=True, key='t_impots_hist_pm2')
                        except Exception as e_hist_impots: st.error(f"Erreur histogrammes (Impôts): {e_hist_impots}"); logging.error(f"Err histos Impots: {e_hist_impots}", exc_info=True)
                        # STOCKAGE pour Export Zip
                        st.session_state['export_fig_hist_vf_imp_t5'] = fig_hist_vf_imp
                        st.session_state['export_fig_hist_pm2_imp_t5'] = fig_hist_pm2_imp

                        st.subheader("Surface vs Prix (par Nombre de Pièces - Impôts)")
                        fig_scatter_vf_imp = None; fig_scatter_pm2_imp = None # Init
                        try:
                            filtered_df_impots_scatter = filtered_df_impots.copy()
                            if 'nombre_pieces_principales' in filtered_df_impots_scatter.columns:
                                if 'categorize_pieces' in locals() and callable(categorize_pieces):
                                    filtered_df_impots_scatter['nombre_pieces_cat'] = filtered_df_impots_scatter['nombre_pieces_principales'].apply(categorize_pieces)
                                    piece_order_imp = ["=1", "=2", "=3", "=4", ">4", "Autre/Inconnu"]
                                    valid_piece_cats_imp = [cat for cat in piece_order_imp if cat in filtered_df_impots_scatter['nombre_pieces_cat'].unique()]
                                    imp_col_scatter1, imp_col_scatter2 = st.columns(2)
                                    with imp_col_scatter1:
                                        fig_scatter_vf_imp = px.scatter(filtered_df_impots_scatter, x="surface_reelle_bati", y="valeur_fonciere", color="nombre_pieces_cat", title="Surface vs VF", labels={"surface_reelle_bati": "Surface (m²)", "valeur_fonciere": "Valeur (€)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats_imp}, hover_data=['nom_commune','adresse_brute'], template="plotly_white") # Ajout template
                                        st.plotly_chart(fig_scatter_vf_imp, use_container_width=True, key='t_impots_scatter_vf')
                                    with imp_col_scatter2:
                                        fig_scatter_pm2_imp = px.scatter(filtered_df_impots_scatter, x="surface_reelle_bati", y="prix_m2", color="nombre_pieces_cat", title="Surface vs Prix m²", labels={"surface_reelle_bati": "Surface (m²)", "prix_m2": "Prix m² (€/m²)", "nombre_pieces_cat": "Nb Pièces"}, category_orders={"nombre_pieces_cat": valid_piece_cats_imp}, hover_data=['nom_commune','adresse_brute'], template="plotly_white") # Ajout template
                                        st.plotly_chart(fig_scatter_pm2_imp, use_container_width=True, key='t_impots_scatter_pm2')
                                else: st.error("Fonction 'categorize_pieces' non définie.")
                            else: st.warning("Col 'nombre_pieces_principales' non trouvée pour nuages (Impôts).")
                        except Exception as e_scatter_impots: st.error(f"Erreur nuages de points (Impôts): {e_scatter_impots}"); logging.error(f"Err nuages points Impots: {e_scatter_impots}", exc_info=True)
                        # STOCKAGE pour Export Zip
                        st.session_state['export_fig_scatter_vf_imp_t5'] = fig_scatter_vf_imp
                        st.session_state['export_fig_scatter_pm2_imp_t5'] = fig_scatter_pm2_imp

                        # --- Affichage Table Détaillée (Optionnel) ---
                        with st.expander("Voir les données détaillées (Impôts)"):
                            # ... (Code affichage table détaillée inchangé) ...
                            cols_to_display_impots = ['nom_commune', 'adresse_brute', 'date_mutation', 'annee_construction', 'valeur_fonciere', 'prix_m2', 'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain', 'Ref enreg']
                            cols_present = [col for col in cols_to_display_impots if col in filtered_df_impots.columns]
                            if cols_present: st.dataframe(filtered_df_impots[cols_present], use_container_width=True, hide_index=True)
                            else: st.warning("Aucune colonne sélectionnée à afficher.")

                        # --- Bouton Download Onglet Impôts ---
                        # ... (Code bouton download CSV inchangé) ...
                        try:
                            if 'cols_present' in locals() and cols_present and not filtered_df_impots.empty:
                                csv_impots = filtered_df_impots[cols_present].to_csv(index=False, encoding='utf-8-sig', sep=';')
                                st.download_button(label="Télécharger les résultats filtrés (Données Impôts)", data=csv_impots, file_name="resultats_donnees_impots_filtrees.csv", mime="text/csv", key='download_tab_impots')
                        except Exception as e_dl_impots: st.error(f"Erreur préparation téléchargement (Impôts): {e_dl_impots}")

                    # Fin de 'if nombre_biens_impots > 0:'
                    else:
                        st.info("Aucune donnée Impôts à afficher avec les filtres actuels.")
                        # Reset des clés pour l'export si pas de données filtrées
                        for key in keys_to_reset_t5: st.session_state[key] = None

                # Fin de 'if not impots_data_available: ... elif ... else:'
            # Fin de 'with tab5:'


                # Fin de la condition 'if not impots_data_available: ... elif ... else:'
            # Fin de with tab_impots:

            # ===========================================
            # ========= NOUVEL ONGLET: Synthèse =========
            # ===========================================
            # ======================================================
            # ========= ONGLET 6: Synthèse Comparative ========= (Basé sur temp.txt + Stockage Zip PNG)
            # ======================================================
            # ======================================================
            # ========= ONGLET 6: Synthèse Comparative ========= (Version Finale avec Stockage Zip PNG)
            # ======================================================
            with tab6:
                # Assurer imports nécessaires au début du script principal
                # import pandas as pd
                # import numpy as np
                # import streamlit as st
                # import plotly.express as px
                # import matplotlib.pyplot as plt # Pour KDE
                # import seaborn as sns # Pour KDE
                # import logging

                st.header("💡 Synthèse Comparative des Marchés")
                st.caption("Cette section résume les données des autres onglets en fonction des filtres qui y sont appliqués.")

                # --- A. Récupération des données depuis l'état de session ---
                # Utiliser la clé correcte ('export_df_filtered_t1') comme corrigé
                df_dvf_t1 = st.session_state.get('export_df_filtered_t1', pd.DataFrame())
                df_ancien_t3 = st.session_state.get('filtered_df_tab3_ancien', pd.DataFrame())
                df_neuf_t3 = st.session_state.get('filtered_df_tab3_neuf', pd.DataFrame())
                df_impots_t5 = st.session_state.get('filtered_df_tab5', pd.DataFrame())

                # Préparer la liste des sources et le DF combiné pour les graphiques
                sources_labels = {
                    "Vendus (DVF)": df_dvf_t1,
                    "À Vendre (Ancien)": df_ancien_t3,
                    "À Vendre (Neuf)": df_neuf_t3,
                    "Vendus (Impôts)": df_impots_t5

                #     # Préparer la liste des sources et le DF combiné pour les graphiques
                # sources_labels = {
                #     "Vendu (DVF - Filtres Onglet1)": df_dvf_t1,
                #     "À Vendre (Ancien - Filtres Onglet3)": df_ancien_t3,
                #     "À Vendre (Neuf - Filtres Onglet3)": df_neuf_t3,
                #     "Vendu (Impôts - Filtres Onglet5)": df_impots_t5
                }
                dfs_to_combine_synthese = []
                for label, df_source in sources_labels.items():
                    if df_source is not None and not df_source.empty and 'prix_m2' in df_source.columns:
                        temp_df = df_source[['prix_m2']].dropna().copy()
                        temp_df['prix_m2'] = pd.to_numeric(temp_df['prix_m2'], errors='coerce')
                        temp_df = temp_df[temp_df['prix_m2'] > 0]
                        if not temp_df.empty:
                            temp_df['Source'] = label
                            dfs_to_combine_synthese.append(temp_df)

                data_available_for_plot = len(dfs_to_combine_synthese) > 0
                df_hist_synthese = pd.DataFrame() # Initialiser

                # --- Initialiser/Reset clés session_state pour cet onglet ---
                keys_to_reset_t6 = [k for k in export_keys if k.endswith('_t6')]
                if not data_available_for_plot:
                    st.warning("Aucune donnée de prix/m² valide disponible depuis les autres onglets pour la synthèse. Veuillez appliquer des filtres pertinents.")
                    for key in keys_to_reset_t6: st.session_state[key] = None # Reset si pas de données
                else:
                    df_hist_synthese = pd.concat(dfs_to_combine_synthese, ignore_index=True)

                    # --- B. Tableau Comparatif Simplifié (Prix/m² Moyen) ---
                    st.subheader("Comparaison des Prix au m² Moyens")

                    # Helper function (définie localement dans temp.txt)
                    def calculate_synthese_stats(df, label):
                        stats = {'Source': label, 'Nb Biens': 0, 'Prix/m² Moyen': 'N/A', 'Prix/m² Moyen Num': np.nan, 'Prix/m² Median Num': np.nan}
                        if df is not None and not df.empty and 'prix_m2' in df.columns:
                            stats['Nb Biens'] = len(df) # Compte total basé sur le DF source
                            valid_prices = pd.to_numeric(df['prix_m2'], errors='coerce').dropna()
                            valid_prices = valid_prices[valid_prices > 0]
                            if not valid_prices.empty:
                                mean_val = valid_prices.mean()
                                median_val = valid_prices.median()
                                stats['Prix/m² Moyen'] = f"{int(round(mean_val)):,} €/m²".replace(',', ' ')
                                stats['Prix/m² Moyen Num'] = mean_val
                                stats['Prix/m² Median Num'] = median_val
                        return stats

                    # Calculer les stats
                    dict_dvf_s = calculate_synthese_stats(df_dvf_t1, "Vendu (DVF)")
                    dict_ancien_s = calculate_synthese_stats(df_ancien_t3, "À Vendre (Ancien)")
                    dict_neuf_s = calculate_synthese_stats(df_neuf_t3, "À Vendre (Neuf)")
                    dict_impots_s = calculate_synthese_stats(df_impots_t5, "Vendu (Impôts)")

                    stats_list_synthese = [d for d in [dict_dvf_s, dict_ancien_s, dict_neuf_s, dict_impots_s] if d and d.get('Nb Biens', 0) > 0]

                    df_stats_synthese = pd.DataFrame() # Init
                    if stats_list_synthese:
                        df_stats_synthese = pd.DataFrame(stats_list_synthese).set_index('Source')

                        # Calcul Diff % vs DVF T1
                        df_stats_synthese['Diff % vs DVF Onglet1'] = 'N/A'
                        dvf_source_label = "Vendu (DVF)"
                        if dvf_source_label in df_stats_synthese.index:
                            dvf_avg_price_num = df_stats_synthese.loc[dvf_source_label, 'Prix/m² Moyen Num']
                            if pd.notna(dvf_avg_price_num) and dvf_avg_price_num > 0:
                                for idx in df_stats_synthese.index:
                                    if idx != dvf_source_label:
                                        current_avg_num = df_stats_synthese.loc[idx, 'Prix/m² Moyen Num']
                                        if pd.notna(current_avg_num):
                                            diff_percent = ((current_avg_num - dvf_avg_price_num) / dvf_avg_price_num) * 100
                                            df_stats_synthese.loc[idx, 'Diff % vs DVF Onglet1'] = f"{diff_percent:+.1f}%"
                            else: st.caption("Calcul Diff % impossible (Moyenne DVF invalide ou nulle).")
                        else: st.caption("Calcul Diff % impossible (Source DVF Onglet 1 absente ou vide).")

                        # Afficher tableau simplifié et Stocker
                        cols_to_display_synthese = ['Nb Biens', 'Prix/m² Moyen', 'Diff % vs DVF Onglet1']
                        cols_present_synthese = [c for c in cols_to_display_synthese if c in df_stats_synthese.columns]
                        st.dataframe(df_stats_synthese[cols_present_synthese])
                        # --- STOCKAGE pour Export Zip ---
                        st.session_state['export_df_stats_synthese_t6'] = df_stats_synthese[cols_present_synthese].copy()
                        logging.info(f"Tab 6: df_stats_synthese stocké.")
                        # --- FIN STOCKAGE ---
                    else:
                        st.warning("Aucune source de données valide trouvée pour la synthèse.")
                        st.session_state['export_df_stats_synthese_t6'] = None # Reset
                    # --- Fin Tableau ---

                    st.markdown("---") # Séparateur visuel

                    # --- C. Histogramme Comparatif Étendu ---
                    st.subheader("Distribution Comparée (Histogramme)")
                    st.caption("Superposition des distributions pour visualiser le recouvrement des prix.")
                    fig_hist_synthese = None # Init figure
                    try:
                        fig_hist_synthese = px.histogram(
                            df_hist_synthese,
                            x="prix_m2", color="Source", barmode='group',
                            marginal="box", opacity=0.6, title="Distribution des Prix au m²",
                            nbins=50, labels={'prix_m2': 'Prix au m² (€/m²)', 'Source': 'Source des Données'},
                            template="plotly_white" # <-- AJOUT TEMPLATE
                        )
                        fig_hist_synthese.update_layout(yaxis_title="Nombre de Biens", bargap=0.1)
                        st.plotly_chart(fig_hist_synthese, use_container_width=True, key="t6_hist_compare")
                    except Exception as e_hist_synt:
                        st.error(f"Erreur histogramme comparatif synthèse: {e_hist_synt}")
                        logging.error(f"Err hist comp T6: {e_hist_synt}", exc_info=True)
                    # --- STOCKAGE pour Export Zip ---
                    st.session_state['export_fig_hist_synthese_t6'] = fig_hist_synthese
                    # --- FIN STOCKAGE ---
                    # --- Fin Histogramme ---

                    st.markdown("---")

                    # --- D. Box Plots ---
                    st.subheader("Comparaison par Box Plots")
                    st.caption("Visualise médiane, quartiles, étendue et valeurs atypiques pour chaque source.")
                    col_prix_perso1, col_prix_perso2 = st.columns(2)
                    with col_prix_perso1: prix_achat_perso = st.number_input("Votre Prix/m² Achat (€/m²):", min_value=0, value=0, step=50, format="%d", key="t6_prix_achat_perso", help="Entrez votre prix d'achat au m² pour le visualiser. Laissez à 0 pour ne pas l'afficher.")
                    with col_prix_perso2: prix_vente_perso = st.number_input("Votre Prix/m² Vente Cible (€/m²):", min_value=0, value=0, step=50, format="%d", key="t6_prix_vente_perso", help="Entrez votre prix de vente cible au m². Laissez à 0 pour ne pas l'afficher.")

                    fig_box_synthese = None # Init figure
                    try:
                        fig_box_synthese = px.box(
                            df_hist_synthese,
                            x="Source", y="prix_m2", color="Source", points=False,
                            labels={'prix_m2': 'Prix au m² (€/m²)', 'Source': ''},
                            title="Comparaison Box Plots Prix/m²",
                            template="plotly_white" # <-- AJOUT TEMPLATE
                        )
                        fig_box_synthese.update_layout(xaxis_title=None)
                        if prix_achat_perso > 0: fig_box_synthese.add_hline(y=prix_achat_perso, line_dash="dot", line_color="green", annotation_text=f"Achat: {prix_achat_perso} €/m²", annotation_position="bottom left", annotation_font_color="green", layer='below')
                        if prix_vente_perso > 0: fig_box_synthese.add_hline(y=prix_vente_perso, line_dash="dash", line_color="red", annotation_text=f"Vente: {prix_vente_perso} €/m²", annotation_position="top right", annotation_font_color="red", layer='below')
                        st.plotly_chart(fig_box_synthese, use_container_width=True, key="t6_box_compare")
                    except Exception as e_box_synt:
                        st.error(f"Erreur lors de la création des Box Plots : {e_box_synt}")
                        logging.error(f"Err box plot T6: {e_box_synt}", exc_info=True)
                    # --- STOCKAGE pour Export Zip ---
                    st.session_state['export_fig_box_synthese_t6'] = fig_box_synthese
                    # --- FIN STOCKAGE ---
                    # --- Fin Box Plots ---

                    st.markdown("---")

                    # --- E. Comparaison Moyennes/Médianes (Bar Chart) ---
                    st.subheader("Comparaison des Moyennes / Médianes (Barres)")
                    st.caption("Compare les indicateurs centraux (basés sur les prix > 0) pour chaque source.")
                    fig_bar_stats = None # Init figure
                    if not df_stats_synthese.empty:
                        try:
                            stats_for_bar = df_stats_synthese[['Prix/m² Moyen Num', 'Prix/m² Median Num']].copy()
                            stats_for_bar.columns = ['Moyenne', 'Médiane']
                            stats_for_bar = stats_for_bar.reset_index()
                            stats_melted = stats_for_bar.melt(id_vars='Source', var_name='Statistique', value_name='Prix/m² (€/m²)')
                            stats_melted.dropna(subset=['Prix/m² (€/m²)'], inplace=True)

                            if not stats_melted.empty:
                                fig_bar_stats = px.bar(
                                    stats_melted, x='Source', y='Prix/m² (€/m²)', color='Statistique',
                                    barmode='group', text_auto='.0f', title="Prix/m² Moyen et Médian par Source",
                                    template="plotly_white" # <-- AJOUT TEMPLATE
                                )
                                fig_bar_stats.update_traces(textposition='outside')
                                fig_bar_stats.update_layout(xaxis_title=None, yaxis_title="Prix/m² (€/m²)")
                                st.plotly_chart(fig_bar_stats, use_container_width=True, key="t6_bar_stats")
                            else: st.info("Aucune statistique de moyenne/médiane à afficher en barres.")
                        except Exception as e_bar_stats:
                            st.error(f"Erreur lors de la création du graphique en barres des statistiques : {e_bar_stats}")
                            logging.error(f"Err bar chart stats T6: {e_bar_stats}", exc_info=True)
                    else:
                        st.info("Tableau de statistiques vide, impossible de générer le graphique en barres.")
                    # --- STOCKAGE pour Export Zip ---
                    st.session_state['export_fig_bar_stats_t6'] = fig_bar_stats
                    # --- FIN STOCKAGE ---
                    # --- Fin Bar Chart Stats ---

                    st.markdown("---")

                    # --- F. Graphes de Densité KDE ---
                    st.subheader("Comparaison par Densité (KDE)")
                    st.caption("Montre la forme estimée de la distribution des prix pour chaque source.")
                    fig_kde = None # Init figure
                    if SEABORN_MPL_AVAILABLE:
                        try:
                            fig_kde, ax_kde = plt.subplots(figsize=(10, 6))
                            sns.kdeplot(data=df_hist_synthese, x='prix_m2', hue='Source', fill=True, common_norm=False, alpha=0.4, linewidth=1.5, ax=ax_kde)
                            ax_kde.set_title('Estimation de Densité (KDE) par Source')
                            ax_kde.set_xlabel('Prix au m² (€/m²)')
                            ax_kde.set_ylabel('Densité')
                            fig_kde.tight_layout() # Ajustement automatique
                            st.pyplot(fig_kde)
                        except Exception as e_kde_synt:
                            st.error(f"Erreur lors de la création du graphique KDE : {e_kde_synt}")
                            logging.error(f"Err KDE plot T6: {e_kde_synt}", exc_info=True)
                    else:
                         st.info("Graphique KDE non disponible car les bibliothèques Seaborn et/ou Matplotlib n'ont pas pu être importées.")
                    # --- STOCKAGE pour Export Zip ---
                    # Utiliser export_fig_kde_t6 comme défini dans export_keys
                    st.session_state['export_fig_kde_t6'] = fig_kde # Stocke la figure Matplotlib
                    # --- FIN STOCKAGE ---
                    # --- Fin KDE ---

                # --- Fin de la condition if not data_available_for_plot: else: ---
            # --- Fin de with tab6 ---


                # --- Fin de la condition if not data_available_for_plot: else: ---
            # --- Fin de with tab6 ---

            # --- Fin de with tab6 ---
    # --- Fin de with tab6 ---

    # ... (conditions else pour chargement DVF, etc. à la fin du script) ... [620-621]




                # --- Fin de la condition if df_final_geo_filtered.empty: else: ---
            # --- Fin Contenu Principal ---
      #  elif not selected_communes: 
      #      st.info("⬅️ Veuillez sélectionner au moins une commune...")
      #  elif not selected_sections: 
      #      st.info("⬅️ Veuillez sélectionner au moins une section cadastrale...")
    # --- Fin de la condition if data_loaded and not df_base.empty: ---
    elif st.session_state.data_loaded == False and uploaded_file_dvf is not None: 
        st.error("Le chargement a échoué...")
    # --- Fin de la condition if uploaded_file is not None: ---
    else:
        st.warning("⬅️ Veuillez charger un fichier CSV DVF pour commencer...")

    # --- Fin de l'application ---
else:
    # Si la connexion échoue ou n'est pas faite, on arrête l'exécution ici.
    st.stop()