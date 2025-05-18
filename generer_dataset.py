# -*- coding: utf-8 -*-
"""
Script pour transformer un dataset de prédicteurs généré par Mockaroo
en ajoutant des variables cibles (Final_Exam_Score, Passed_Course)
corrélées et avec du bruit, puis l'exporter en CSV.
"""

import pandas as pd
import numpy as np
import os # Pour vérifier l'existence du fichier

# --- Configuration ---

# Le nom du fichier CSV que tu as téléchargé depuis Mockaroo.
# Il doit contenir les colonnes : Hours_Studied, Previous_Score, Attendance_Rate
input_filename = 'mockaroo_predicteurs.csv'

# Le nom du fichier CSV que ce script va créer avec les colonnes cibles.
# C'est ce fichier que tu utiliseras dans ton notebook Jupyter.
output_filename = 'dataset_etudiant_corrélé.csv'

# Délimiteur utilisé DANS LE FICHIER CSV généré par Mockaroo
# D'après ton image, c'était le point-virgule ';'. Adapte si Mockaroo change.
input_delimiter = ','

# Délimiteur souhaité pour le fichier CSV de sortie (la virgule ',' est standard)
output_delimiter = ','

# Paramètres pour la génération des variables cibles corrélées
# Ces paramètres définissent la "force" de la corrélation et le bruit
base_score = 8.0      # On ajuste légèrement la base
weight_hours = 0.6    # AUGMENTÉ : Pour que les heures étudiées aient plus d'impact direct
weight_prev = 0.15    # DIMINUÉ : Réduit l'influence de la note précédente
weight_attendance = 0.02 # DIMINUÉ : Réduit l'influence de la présence
bruit_amplitude = 1.5 # DIMINUÉ : Réduit la dispersion aléatoire des points

# Note seuil pour déterminer si l'étudiant a réussi ou non (garde la même)
note_passage = 15.0 # Nouvelle Note seuil (exemple)

# --- Vérifier et Charger les données depuis le fichier Mockaroo ---

print(f"Tentative de chargement des données depuis '{input_filename}'...")

if not os.path.exists(input_filename):
    print(f"Erreur : Le fichier d'entrée '{input_filename}' est introuvable.")
    print("Assurez-vous que le fichier CSV de Mockaroo (contenant les prédicteurs) est dans le même répertoire que ce script.")
    exit() # Quitte le script si le fichier n'existe pas

try:
    # Charger le dataset en spécifiant le délimiteur correct
    data = pd.read_csv(input_filename, sep=input_delimiter)
    print(f"Fichier '{input_filename}' chargé avec succès.")
    print("\nPremières lignes du dataset chargé :")
    print(data.head())
    print("\nInformations sur les colonnes :")
    data.info()

except Exception as e:
    print(f"Erreur lors du chargement du fichier '{input_filename}' : {e}")
    print("Vérifiez le nom du fichier et le délimiteur configuré dans le script (input_delimiter).")
    exit()

# --- Vérifier que les colonnes prédicteurs nécessaires sont présentes ---
required_predictors = ['Hours_Studied', 'Previous_Score', 'Attendance_Rate']
if not all(col in data.columns for col in required_predictors):
    missing = [col for col in required_predictors if col not in data.columns]
    print(f"\nErreur : Les colonnes prédicteurs requises sont manquantes dans le fichier chargé : {missing}")
    print("Assurez-vous que le fichier CSV de Mockaroo contient ces colonnes avec les bons noms.")
    exit()


# --- Générer les variables cibles corrélées en Python ---

print("\nGénération des variables cibles (Final_Exam_Score, Passed_Course)...")

# 1. Calculer le score brut en utilisant la formule et les prédicteurs
# np.random.rand(len(data)) génère un tableau de nombres aléatoires (bruit) entre 0 et 1
bruit_aleatoire = (np.random.rand(len(data)) - 0.5) * bruit_amplitude

# Utilise les opérations vectorielles de pandas/numpy pour calculer rapidement sur toutes les lignes
data['Final_Exam_Score_brut'] = base_score + \
                                (weight_hours * data['Hours_Studied']) + \
                                (weight_prev * data['Previous_Score']) + \
                                (weight_attendance * data['Attendance_Rate']) + \
                                bruit_aleatoire

# 2. Appliquer les bornes [0.0, 20.0] au score final
# La méthode .clip(lower, upper) est parfaite pour ça dans pandas
data['Final_Exam_Score'] = data['Final_Exam_Score_brut'].clip(lower=0.0, upper=20.0)

# 3. Créer la colonne binaire Passed_Course (1 si >= note_passage, 0 sinon)
# La comparaison (data['Final_Exam_Score'] >= note_passage) retourne une série de True/False
# .astype(int) convertit True en 1 et False en 0
data['Passed_Course'] = (data['Final_Exam_Score'] >= note_passage).astype(int)

# Supprimer la colonne brute temporaire si tu ne veux pas la garder
data = data.drop(columns=['Final_Exam_Score_brut'])

print("Variables cibles générées et ajoutées au dataset.")
print("Aperçu du dataset modifié :")
print(data.head())


# --- Exporter le dataset transformé vers un nouveau fichier CSV ---

print(f"\nExportation du dataset transformé vers '{output_filename}'...")

try:
    # Exporter en CSV standard (virgule comme délimiteur)
    # index=False empêche pandas d'écrire l'index de la ligne comme une colonne
    data.to_csv(output_filename, sep=output_delimiter, index=False)
    print(f"Dataset exporté avec succès vers '{output_filename}'.")
    print(f"Tu peux maintenant utiliser ce fichier dans ton notebook Jupyter pour le TP.")

except Exception as e:
    print(f"Erreur lors de l'exportation du fichier : {e}")
    print("Vérifiez que le script a les permissions d'écrire dans le répertoire.")

print("\nScript terminé.")