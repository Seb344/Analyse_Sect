# -*- coding: utf-8 -*-
# pdf_processor.py

import camelot
import pandas as pd
import numpy as np
import re
import logging
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONSTANTES ---
# COLUMN_MAPPING MIS À JOUR pour inclure Appartements et Prix ( E )
COLUMN_MAPPING = {
    'Ref enreg': 'Ref enreg',
    'Dept': 'dept',                   # Sera appliqué si présent (appartements)
    'Commune': 'nom_commune',
    'Adresse': 'adresse_brute',
    'Date vente': 'date_mutation',
    'Année construct': 'annee_construction',
    'Nbre pièces': 'nombre_pieces_principales',
    'Prix ( E )': 'valeur_fonciere',  # <- IMPORTANT: doit être généré correctement
    'Surface terrain': 'surface_terrain',    # Présent pour maisons, peut manquer pour apparts
    'Surface utile': 'surface_reelle_bati',
    'Etage': 'etage',                   # Pour appartements
    'Surface carrez': 'surface_carrez',       # Pour appartements
    # Ignorer les Prix/m² pré-calculés
    # 'Prix/m² surfcarrez': 'prix_m2_carrez_pdf',
    # 'Prix/m² surfutile': 'prix_m2_utile_pdf'
}

# --- FONCTIONS UTILITAIRES ---
# clean_numeric_value (INCHANGÉ)
def clean_numeric_value(value_str):
    if pd.isna(value_str) or value_str == '': return np.nan
    cleaned = str(value_str)
    cleaned = re.sub(r'\s+', '', cleaned)
    cleaned = cleaned.replace('€', '')
    if '.' in cleaned and ',' in cleaned:
        if cleaned.find('.') < cleaned.find(','):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else: cleaned = cleaned.replace(',', '')
    elif '.' in cleaned:
         if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
    elif ',' in cleaned: cleaned = cleaned.replace(',', '.')
    try: return float(cleaned)
    except ValueError:
        logging.debug(f"CleanNum Failed: '{value_str}' -> '{cleaned}'")
        return np.nan

# combine_headers SIMPLIFIÉ : combine et nettoie juste les noms de base
def combine_headers(row0, row1):
    """Combine les deux lignes d'en-tête. Nettoyage MINIMAL."""
    headers = []
    num_cols = min(len(row0), len(row1))
    logging.debug("--- Entering combine_headers (simplified) ---")
    for i in range(num_cols):
        h0 = str(row0.iloc[i]).strip() if i < len(row0) and pd.notna(row0.iloc[i]) else ""
        h1 = str(row1.iloc[i]).strip() if i < len(row1) and pd.notna(row1.iloc[i]) else ""
        combined = f"{h0} {h1}".strip()
        # Appliquer seulement les corrections non ambiguës
        if combined == "Datevente": combined = "Date vente"
        # Ne plus essayer de deviner 'Prix ( E )' ou 'Titre_Prix...' ici
        headers.append(combined)
        logging.debug(f"Combine S - Col {i}: Raw Combined = '{combined}'")
    logging.debug(f"Combine S - Output Headers: {headers}")
    logging.debug("--- Exiting combine_headers (simplified) ---")
    return headers

# --- _process_extracted_data (Logique Conditionnelle) ---
def _process_extracted_data(df_raw, type_bien): # Accepte type_bien
    """
    Traite le DataFrame brut. Logique conditionnelle pour maisons/appartements.
    """
    global COLUMN_MAPPING

    if df_raw.shape[0] < 2:
        logging.error("PROCESS: Pas assez de lignes pour traiter les en-têtes.")
        return pd.DataFrame()

    # --- 1. Combiner les En-têtes (Version Simplifiée) ---
    logging.info(f"PROCESS: Combinaison simplifiée des en-têtes pour type '{type_bien}'...")
    # Headers bruts combinés (ex: contient 'Prix()', '', 'Dept'...)
    raw_combined_headers = combine_headers(df_raw.iloc[0], df_raw.iloc[1])
    logging.info(f"PROCESS: Raw Combined Headers: {raw_combined_headers}")

    # --- 2. Extraire les données ---
    df = df_raw[2:].copy()
    logging.debug(f"PROCESS: Shape données brutes (df): {df.shape}")
    num_data_cols = len(df.columns)
    num_raw_headers = len(raw_combined_headers)

    # --- 3. Logique Conditionnelle basée sur type_bien ---
    final_headers = [] # La liste finale des headers à appliquer
    cols_to_drop_indices = [] # Indices des colonnes de DONNÉES à supprimer

    if type_bien == 'maison':
        logging.info("PROCESS: Application logique spécifique 'maison'...")
        # Identifier l'index du titre 'Prix()' et de la valeur ''
        try:
            # L'index dont la DONNÉE doit être supprimée
            index_prix_titre = raw_combined_headers.index('Prix()') # Devrait être 8
            cols_to_drop_indices.append(index_prix_titre)
            logging.info(f"PROCESS (maison): Marqué index {index_prix_titre} ('Prix()') pour suppression de données.")
            # L'index de la valeur prix (header vide)
            index_prix_valeur = raw_combined_headers.index('') # Devrait être 9
            raw_combined_headers[index_prix_valeur] = 'Prix ( E )' # Renommer le header vide en 'Prix ( E )'
            logging.info(f"PROCESS (maison): Renommé header à l'index {index_prix_valeur} en 'Prix ( E )'.")
        except ValueError:
            logging.error("PROCESS (maison): Impossible de trouver 'Prix()' ou '' dans les headers bruts.")
            return pd.DataFrame()

        # Identifier l'index de 'Dept'
        try:
            index_dept = raw_combined_headers.index('Dept') # Devrait être 2
            # Marquer la colonne de données 'Dept' pour suppression
            cols_to_drop_indices.append(index_dept)
            logging.info(f"PROCESS (maison): Marqué index {index_dept} ('Dept') pour suppression de données.")
            # Supprimer 'Dept' de la liste finale des headers
            final_headers = [h for h in raw_combined_headers if h != 'Dept']
        except ValueError:
            logging.warning("PROCESS (maison): Header 'Dept' non trouvé, suppression ignorée.")
            final_headers = list(raw_combined_headers) # Garder tous les autres headers

        # Supprimer aussi le header 'Prix()' qui a été marqué pour suppression de données
        final_headers = [h for h in final_headers if h != 'Prix()']

    elif type_bien == 'appartement':
        logging.info("PROCESS: Application logique spécifique 'appartement'...")
        # Identifier l'index de 'Prix()' (qui contient la valeur pour apparts)
        try:
            index_prix_valeur = raw_combined_headers.index('Prix()') # Devrait être 11
            # Renommer cet header en 'Prix ( E )'
            raw_combined_headers[index_prix_valeur] = 'Prix ( E )'
            logging.info(f"PROCESS (appart): Renommé header à l'index {index_prix_valeur} ('Prix()') en 'Prix ( E )'.")
            # Aucune colonne de données à supprimer pour le prix ici
        except ValueError:
            logging.error("PROCESS (appart): Impossible de trouver 'Prix()' dans les headers bruts.")
            return pd.DataFrame()
        # Aucune colonne à supprimer pour 'Dept'
        final_headers = list(raw_combined_headers) # Garder tous les headers (avec 'Prix ( E )' renommé)

    else:
        logging.error(f"PROCESS: type_bien '{type_bien}' non reconnu. Arrêt.")
        return pd.DataFrame()

    # --- 4. Supprimer les Colonnes de Données Marquées ---
    cols_to_drop_indices = sorted(list(set(cols_to_drop_indices)), reverse=True) # Trier pour éviter erreurs d'index
    logging.info(f"PROCESS: Indices de données à supprimer: {cols_to_drop_indices}")
    initial_data_cols = len(df.columns)
    for index_to_drop in cols_to_drop_indices:
        if index_to_drop < initial_data_cols : # Vérifier si l'index existe avant drop
            try:
                col_name = df.columns[index_to_drop] # Nom numérique avant renommage
                df = df.drop(columns=[col_name])
                logging.info(f"PROCESS: Colonne de données à l'index {index_to_drop} (nom brut: {col_name}) supprimée.")
            except Exception as e_drop:
                logging.error(f"PROCESS: Erreur lors de la suppression de la colonne de données à l'index {index_to_drop}: {e_drop}")
                return pd.DataFrame()
        else:
            logging.warning(f"PROCESS: Index de données {index_to_drop} à supprimer est hors limites (Nb cols: {initial_data_cols}). Suppression ignorée.")

    logging.info(f"PROCESS: Shape données après suppressions conditionnelles: {df.shape}")

    # --- 5. Appliquer les En-têtes Finaux ---
    num_final_headers = len(final_headers)
    num_final_data_cols = len(df.columns)
    logging.info(f"PROCESS: Vérification dimensions finales: len(headers)={num_final_headers}, len(cols)={num_final_data_cols}")

    if num_final_headers == num_final_data_cols:
        df.columns = final_headers
        logging.info(f"PROCESS: En-têtes finaux ({num_final_headers}) appliqués aux données ({num_final_data_cols}).")
    else:
        logging.error(f"PROCESS: Discordance finale de dimensions non résolue ({num_final_headers} vs {num_final_data_cols}). Arrêt.")
        logging.error(f"Headers: {final_headers}")
        logging.error(f"Colonnes Data: {df.columns.tolist()}") # Lister les noms numériques
        return pd.DataFrame()

    df.reset_index(drop=True, inplace=True)
    logging.debug(f"PROCESS: Données après application en-têtes finaux (head):\n{df.head().to_string()}")

    # --- 6. Filtrer les lignes d'en-tête restantes ---
    # (Logique inchangée)
    logging.info("PROCESS: Filtrage des lignes d'en-tête potentielles restantes...")
    # ... (code identique) ...
    if 'Ref enreg' in df.columns:
        header_pattern = r'^(Ref|enreg)$'
        is_header_row = df['Ref enreg'].astype(str).str.strip().str.match(header_pattern, case=False, na=False)
        count_removed = is_header_row.sum()
        if count_removed > 0:
            df = df[~is_header_row].copy()
            logging.info(f"PROCESS: {count_removed} lignes d'en-tête potentielles supprimées.")
        df.reset_index(drop=True, inplace=True)

    # --- 7. Fusionner les Lignes Multiples ---
    # (Logique inchangée, basée sur 'Commune')
    logging.info("PROCESS: Tentative de fusion des lignes multiples (v16 - basé sur 'Commune')...")
    # ... (code identique) ...
    primary_row_indicator_col = 'Commune'
    df_merged = pd.DataFrame()
    if primary_row_indicator_col in df.columns:
        try:
            is_primary = df[primary_row_indicator_col].fillna('').astype(str).str.strip() != ''
            df['sale_id'] = is_primary.cumsum()
            agg_funcs = {}
            for col in df.columns:
                if col in ['sale_id', 'RefCad']: continue
                else:
                    agg_funcs[col] = lambda x: x.dropna().astype(str).str.strip().replace('', np.nan).dropna().iloc[0] if not x.dropna().astype(str).str.strip().replace('', np.nan).dropna().empty else ''
            if not agg_funcs: df_merged = df.drop(columns=['sale_id'], errors='ignore')
            else:
                df_merged = df.groupby('sale_id', sort=False, as_index=False).agg(agg_funcs)
                if 'sale_id' in df_merged.columns: df_merged = df_merged.drop(columns=['sale_id'])
            logging.info(f"PROCESS: Données après fusion (v16) - Shape: {df_merged.shape}")
            logging.debug(f"PROCESS: Head après fusion:\n{df_merged.head().to_string()}")
        except Exception as e_merge:
            logging.error(f"PROCESS: ERREUR pendant la fusion (v16): {e_merge}", exc_info=True)
            df_merged = df.drop(columns=['sale_id'], errors='ignore')
            logging.warning("PROCESS: Utilisation des données non fusionnées suite à l'erreur.")
    else:
        logging.warning(f"PROCESS: Colonne indicateur '{primary_row_indicator_col}' non trouvée, fusion impossible.")
        df_merged = df
    if df_merged.empty: return pd.DataFrame()


    # --- 8. Renommage/Mapping Final via COLUMN_MAPPING (mis à jour) ---
    logging.info("PROCESS: Application du mapping final via COLUMN_MAPPING...")
    # ... (logique identique, mais COLUMN_MAPPING est à jour) ...
    current_columns = df_merged.columns
    final_cols_to_rename = {k: v for k, v in COLUMN_MAPPING.items() if k in current_columns}
    cols_to_keep_final = list(final_cols_to_rename.keys())
    missing_final_keys = [k for k in COLUMN_MAPPING.keys() if k not in current_columns]
    if missing_final_keys:
        logging.warning(f"PROCESS: Clés de mapping non trouvées après fusion: {missing_final_keys}")
    cols_to_keep_final = [col for col in cols_to_keep_final if col in df_merged.columns]
    if not cols_to_keep_final:
        logging.error("PROCESS: Aucune colonne à garder après mapping.")
        return pd.DataFrame()
    df_final = df_merged[cols_to_keep_final].copy()
    df_final.rename(columns=final_cols_to_rename, inplace=True)
    logging.info(f"PROCESS: Colonnes après mapping final: {df_final.columns.tolist()}") # Doit contenir valeur_fonciere maintenant
    logging.debug(f"PROCESS: Head après mapping:\n{df_final.head().to_string()}")


    # --- 9. Nettoyage et Conversion Types Finaux ---
    logging.info("PROCESS: Nettoyage final et conversion des types...")
    # (Logique inchangée)
    # ... (code identique) ...
    if 'date_mutation' in df_final.columns: df_final['date_mutation'] = pd.to_datetime(df_final['date_mutation'], errors='coerce', dayfirst=True)
    numeric_cols_to_clean = [
        'nombre_pieces_principales', 'valeur_fonciere', 'surface_terrain',
        'surface_reelle_bati', 'annee_construction', 'etage', 'surface_carrez' # Ajout cols apparts
    ]
    for col in numeric_cols_to_clean:
        if col in df_final.columns:
            df_final[col] = df_final[col].replace('', np.nan).apply(clean_numeric_value)
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    if 'adresse_brute' in df_final.columns: df_final['adresse_brute'] = df_final['adresse_brute'].astype(str).str.strip()

    # --- 10. Calcul Prix/m² ---
    logging.info(f"Calcul de prix_m2 (entier)...")
    # (Logique inchangée)
    # ... (code identique) ...
    final_prix_m2_col_name = 'prix_m2'
    if 'surface_reelle_bati' in df_final.columns and 'valeur_fonciere' in df_final.columns:
        s_bati = pd.to_numeric(df_final['surface_reelle_bati'], errors='coerce')
        v_fonciere = pd.to_numeric(df_final['valeur_fonciere'], errors='coerce')
        if pd.api.types.is_numeric_dtype(s_bati) and pd.api.types.is_numeric_dtype(v_fonciere):
            mask_calcul = (s_bati > 0) & (v_fonciere.notna()) & (s_bati.notna())
            df_final[final_prix_m2_col_name] = np.nan
            calculated_values = v_fonciere[mask_calcul] / s_bati[mask_calcul]
            calculated_values = calculated_values.replace([np.inf, -np.inf], np.nan)
            df_final.loc[mask_calcul, final_prix_m2_col_name] = calculated_values
            try:
                if not np.isfinite(df_final[final_prix_m2_col_name].fillna(0)).all():
                    df_final[final_prix_m2_col_name] = df_final[final_prix_m2_col_name].replace([np.inf, -np.inf], np.nan)
                df_final[final_prix_m2_col_name] = df_final[final_prix_m2_col_name].fillna(0).astype(int)
                logging.info(f"   Colonne '{final_prix_m2_col_name}' calculée et convertie en int.")
            except Exception as e_astype_final:
                logging.error(f"   ERREUR conversion int '{final_prix_m2_col_name}': {e_astype_final}", exc_info=True)
                df_final[final_prix_m2_col_name] = pd.to_numeric(df_final[final_prix_m2_col_name], errors='coerce')
        else:
            logging.error(f"PROCESS: Échec calcul {final_prix_m2_col_name}. Colonnes non numériques.")
            df_final[final_prix_m2_col_name] = 0
    else:
        logging.warning(f"PROCESS: Colonnes manquantes pour calcul {final_prix_m2_col_name}.")
        df_final[final_prix_m2_col_name] = 0

    logging.info("PROCESS: Nettoyage terminé.")
    return df_final

# --- FIN _process_extracted_data ---


# --- FONCTION PRINCIPALE (MODIFIÉE pour passer type_bien) ---
def charger_et_traiter_pdf_ventes(chemin_pdf_ou_objet_fichier, pages_a_lire='2-end'):
    if isinstance(chemin_pdf_ou_objet_fichier, str): display_name = chemin_pdf_ou_objet_fichier
    elif hasattr(chemin_pdf_ou_objet_fichier, 'name'): display_name = f"fichier uploadé '{chemin_pdf_ou_objet_fichier.name}'"
    else: display_name = "objet fichier inconnu"
    logging.info(f"--- Lancement traitement PDF Impôts (Logique Conditionnelle v3): {display_name} ---")
    df_final_result = pd.DataFrame()
    try:
        logging.info(f"Lecture PDF via Camelot (pages: {pages_a_lire})...")
        tables = camelot.read_pdf(
            chemin_pdf_ou_objet_fichier, pages=pages_a_lire, flavor='stream',
            strip_text=' .\n€', edge_tol=500
        )
        logging.info(f"Nombre de tableaux trouvés : {tables.n}")
        if tables.n > 0:
            df_raw_extracted = pd.concat([tbl.df for tbl in tables], ignore_index=True)
            logging.info(f"Extrait brut concaténé - Shape: {df_raw_extracted.shape}")
            # --- Détection type ---
            type_bien = 'inconnu'
            if len(df_raw_extracted) >= 2:
                header_text_line0 = ' '.join(df_raw_extracted.iloc[0].astype(str))
                header_text_line1 = ' '.join(df_raw_extracted.iloc[1].astype(str))
                full_header_text = header_text_line0 + ' ' + header_text_line1
                if 'carrez' in full_header_text.lower(): type_bien = 'appartement'
                else: type_bien = 'maison'
                logging.info(f"Détection : Type '{type_bien}'.")
            else:
                 logging.warning("Moins de 2 lignes extraites, détection type impossible.")
            # --- Fin Détection ---

            # <<< APPEL MODIFIÉ : Passer type_bien >>>
            df_final_result = _process_extracted_data(df_raw_extracted, type_bien)

            if df_final_result.empty: logging.warning("Traitement interne -> DataFrame vide.")
            else: logging.info(f"Traitement interne OK. Shape final: {df_final_result.shape}")
        else: logging.warning("Aucun tableau trouvé par Camelot.")
    except FileNotFoundError: logging.error(f"ERREUR: Fichier PDF non trouvé : {display_name}")
    except ImportError: logging.error("ERREUR CRITIQUE: Camelot/Ghostscript non installés.")
    except Exception as e: logging.error(f"ERREUR inattendue : {e}", exc_info=True)
    logging.info(f"--- Fin traitement PDF Impôts pour {display_name} ---")
    return df_final_result

# --- Bloc Test (inchangé) ---
if __name__ == "__main__":
    print("--- Mode Test du module pdf_processor ---")
    test_pdf_path = r"CHEMIN/VERS/VOTRE/PDF/DE/TEST.pdf"
    pages_test = '2-end'
    if "CHEMIN/VERS/VOTRE/PDF/DE/TEST.pdf" in test_pdf_path:
         print("\n!!! ATTENTION !!! Modifiez 'test_pdf_path' pour tester.")
    else:
        print(f"Test avec le fichier : {test_pdf_path}")
        df_resultat_test = charger_et_traiter_pdf_ventes(test_pdf_path, pages_test)
        if not df_resultat_test.empty:
            print("\n--- Résultat du Test (5 premières lignes) ---")
            print(df_resultat_test.head().to_string())
            print("\n--- Types de données du Test ---")
            print(df_resultat_test.dtypes)
            try:
                output_test_csv = "test_output_pdf_processor.csv"
                df_resultat_test.to_csv(output_test_csv, index=False, sep=';', encoding='utf-8-sig')
                print(f"\nRésultat sauvegardé dans : {output_test_csv}")
            except Exception as e_save_test:
                print(f"\nErreur sauvegarde CSV test: {e_save_test}")
        else:
            print("\nLe test n'a retourné aucune donnée.")

# --- FIN DU FICHIER ---