import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Projet Pompiers")
st.sidebar.title("Sommaire")
pages=["Visualization des donnees", "Regression", "Classification"]
page=st.sidebar.radio("Aller vers", pages)

file_path = r"C:\Users\ruellv\Documents\DataScientest\Incidents_2009_2024.csv"
Incidents=pd.read_csv(file_path)
Mobilisations=pd.read_csv("Mobilisation_2009_2023.csv")
Data_Mergées=pd.read_csv("Data_Mergees.csv")
Data_Encodée=pd.read_csv("Data_Encodee_V2_2_2020.csv")

if page == pages[0] : 
  st.write("### Exploration des Données")
  st.write("Nous avons deux types de dataset à notre disposition: les Incidents Records, contenant 39 colonnes avec des informations sur les incidents ou les pompiers de Londres sont intervenus et les Mobilisation Records, contenant 22 colonnes avec des informations sur le temps de réaction des engins des pompiers.")
  st.dataframe(Incidents.head(10))
  st.dataframe(Mobilisations.head(10))
  st.write("-------")
  
  st.write("Nous démarrons l'exploration des données avec cette representation du nombre d'Incidents par jour de l'année")
  st.write("On remarque un nombre d'interventions comparable chaque année, avec quelques pics ponctuels en 2009, 2013 et 2016, probablement dus à des événements majeurs. Une hausse d'incidents est également observable chaque été.")
  st.image("C:\\Users\\ruellv\\Documents\\DataScientest\\Images Streamlit\\Incidents par jour de l'annee.png")
  st.write("-------")
  
  st.write("Ces graphiques circulaires montrent que la répartition en pourcentage par type d'appel est relativement constante. Près de la moitié des incidents sont des fausses alertes.")
  st.image("C:\\Users\\ruellv\\Documents\\DataScientest\\Images Streamlit\\Typologie d'incidents par annee.png")
  st.write("-------")
  
  st.write("Ce premier graphique montre que la majorité des interventions ont lieu pendant la journée.")
  st.image("C:\\Users\\ruellv\\Documents\\DataScientest\\Images Streamlit\\Nombre d'interventions en fonction de l'heure d'appel.png")
  st.write("Le deuxieme graphique montre le temps de réponse de la première brigade en fonction de l'heure d'appel. On constate des temps de réaction accrus eu petit matin (4h-7h) et en début d'après-midi (12h-17h).")
  st.image("C:\\Users\\ruellv\\Documents\\DataScientest\\Images Streamlit\\Temps moyen de reaction en fonction de l'heure d'appel.png")
  st.write("-------")
  
  st.write("Nous avons commencé a nous pencher plus sur les temps de réaction. nous Montrons ci dessus la distribution des temps de reaction en minutes des premiers et deuxiemes camions intervenus.")
  st.image("C:\\Users\\ruellv\\Documents\\DataScientest\\Images Streamlit\\Distribution du temps de reaction du premier camion sur place.png")
  st.write("-------")
  
  st.write("Les données ont été enrichies avec les localisations des casernes de Londres. Avec les coordonnées des stations et des lieux d'intervention, nous avons pu calculer la distance à parcourir par les camions, une donnée pertinente pour notre analyse.")
  st.image("C:\\Users\\ruellv\\Documents\\DataScientest\\Images Streamlit\\plan des casernes de londres.png")
  st.write("-------")

  st.write("Nous avons ensuite procédé au merge des données Incidents et Mobilisation afin de pouvour travailler sur un seul Dataframe")
  st.dataframe(Data_Mergées.head(10))
  st.write("-------")

  st.write("Nous avons ensuite proédé a l'encodage de nos données")
  st.write("Le Mean Encoding a été appliqué aux données catégorielles qui nous on servis de features.")
  st.write("Nos datasets contiennent des données temporelles sur plusieurs années, mois, jours et heures. Ainsi, le Cyclic Encoding s'est avéré nécessaire..")
  st.dataframe(Data_Encodée.head(10))
  st.text(Data_Encodée.info(memory_usage='deep'))
  st.write("-------")


