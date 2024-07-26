# test de faire comme dans le nb

import streamlit as st
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from feature_engine.encoding import MeanEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error,mean_absolute_error, root_mean_squared_error,r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from imblearn.metrics import macro_averaged_mean_absolute_error ,classification_report_imbalanced, geometric_mean_score , sensitivity_score

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
import xgboost as xgb

import shap
import pickle



# Fonction pour lire l'image et la convertir en base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Utiliser l'image de fond
background_image = "pompiers_londres.jpg"
bg_image_base64 = get_base64_of_bin_file(background_image)

# Définir le style CSS pour la page avec une superposition
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image_base64}");
    background-size: cover;
    background-position: center;
    height: 100vh;
    position: relative;
}}
[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}
.overlay {{
    position: fixed; /* Changer de absolute à fixed */
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.4); /* Couleur noire avec 40% de transparence */
    z-index: 1;
}}
.container {{
    position: relative;
    z-index: 2;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.title {{
    color: white;
    font-size: 3em;
    text-align: center;
    margin: 0;
}}
</style>
"""

# fonction pour charger le df Mobilisation et le df Mobilisation

pages = ["Home","Introduction","Jeu de données","Modélisation", "Conclusion"] 
page = st.sidebar.radio("Allez vers", pages)



# Titre de la page
if page == pages[0] :

    @st.cache_data  
    def Data_Mob(chemin):
        dico = {'IncidentNumber' : 'str',
            'DelayCodeId' : 'object',
            "CalYear" : 'str',
            "ResourceMobilisationId" : "str"}
        Mobilisation = pd.read_csv(chemin, dtype=dico)
        Mobilisation = Mobilisation.drop(["TempsDepartSeconds", "TempsTrajetSeconds"], axis = 1)

        premier_camion = Mobilisation[Mobilisation["PumpOrder"]==1][["IncidentNumber","TurnoutTimeSeconds","TravelTimeSeconds","AttendanceTimeSeconds",
                                                                                "DelayCode_Description","Division_station"]]

        
        second_camion = Mobilisation[Mobilisation["PumpOrder"]==2][["IncidentNumber","TurnoutTimeSeconds","TravelTimeSeconds","AttendanceTimeSeconds",
                                                                                    "DelayCode_Description","Division_station"]]

        premier_camion.rename({"TurnoutTimeSeconds" : "FirstPumpArriving_TurnoutTimeSec",
                    "TravelTimeSeconds" : "FirstPumpArriving_TravelTimeSec",
                                "AttendanceTimeSeconds" : "Mob_FirstPump_AttendanceTime",
                            "DelayCode_Description" : "FirstPump_DelayCode_Description",
                            "Division_station" : "FirstPump_Division_staion"},axis = 1, inplace = True)


        second_camion.rename({"TurnoutTimeSeconds" : "SecondPumpArriving_TurnoutTimeSec",
                            "TravelTimeSeconds" : "SecondPumpArriving_TravelTimeSec",
                            "AttendanceTimeSeconds" : "Mob_SecondPump_AttendanceTime",
                            "DelayCode_Description" : "SecondPump_DelayCode_Description",
                            "Division_station" : 'SecondPump_Division_staion'},axis = 1, inplace = True)
            
        Mob_pompes_12 = premier_camion.merge(second_camion, on = "IncidentNumber", how = "outer")

        return [Mobilisation, Mob_pompes_12]

    # fonction pour charger le df Incident
    @st.cache_data  
    def Data_Incident(chemin):
        dico = {'IncidentNumber' : 'str',
                    "CalYear" : 'str'}
        Incidents = pd.read_csv(chemin, dtype = dico)
        return Incidents

    # fonction pour charger le df mergé
    @st.cache_data  
    def Data_Merge(chemin):
        dico = {'IncidentNumber' : 'str',
                "CalYear" : "str",
                    "Easting_m" : 'str',
                    "Northing_m" : "str",
                    "UPRN" : "str" ,
                    "USRN" : "str",
                    "Easting_rounded" : "str",
                    "Northing_rounded" : "str"
            }
        Merge = pd.read_csv(chemin, dtype = dico)
        return Merge

    # le dataFrame avec les données encodées et filtre 2020
    @st.cache_data
    def Data_Mod(chemin):
        
        dico = {'IncidentNumber' : 'str',     
            }
        df = pd.read_csv(chemin, dtype = dico)
        # ncr de rename les colonnes avec les noms d'origine car les nouvelles étaient utilisées pour conserver les  valeurs originales après un pd.getdummies
        df = df.rename(columns={"IncidentGroup_orig" : "IncidentGroup",
                        "StopCodeDescription_orig" : "StopCodeDescription",
                        "PropertyCategory_orig" : "PropertyCategory"})

        df["dst_StationIncident"] = df["dst_StationIncident"]/1000

        return df

    Mobilisation = Data_Mob("Data/Mobilisation_2009_2023.csv")[0]
    Mob_pompes_12 = Data_Mob("Data/Mobilisation_2009_2023.csv")[1]
    Incidents = Data_Incident("Data/Incidents_2009_2024.csv")
    Merge = Data_Merge("Data/Data_Mergees.csv")
    df_2020 = Data_Mod("Data/Data_Encodee_V2_2_2020.csv") 

    if "Mobilisation" not in st.session_state :
        st.session_state["Mobilisation"] = Data_Mob("Data/Mobilisation_2009_2023.csv")[0]
        
    if "Mob_pompes_12" not in st.session_state :
        st.session_state["Mob_pompes_12"] = Data_Mob("Data/Mobilisation_2009_2023.csv")[1]

    if "Incidents" not in st.session_state :
        st.session_state["Incidents"] = Data_Incident("Data/Incidents_2009_2024.csv")
        
    if "Merge" not in st.session_state :
        st.session_state["Merge"] = Data_Merge("Data/Data_Mergees.csv")
        
    if "df_2020" not in st.session_state :
        st.session_state["df_2020"] = Data_Mod("Data/Data_Encodee_V2_2_2020.csv")

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
    st.markdown('<div class="container"><h1 class="title">Projet Pompiers de la ville de Londres</h1></div>', unsafe_allow_html=True)
 

# Page d'introduction
if page == pages[1] :

    "session state obj",st.session_state
    st.title("Introduction")
    st.markdown(
        """
        **Bienvenue dans le projet "Pompiers de la ville de Londres".**

        La Brigade des Pompiers de Londres, avec 5000 personnes, est le plus grand service
        d'incendie et de secours du pays, protégeant le Grand Londres. Suite aux incendies
        meurtriers passés, des investissements croissants ont été réalisés, le temps de réaction
        étant crucial. Les incendies, souvent débutant par une petite flamme contrôlable,
        se propagent avec le temps, aggravant les conditions de lutte. Le facteur essentiel est
        le temps de réponse des pompiers, crucial pour la survie et la minimisation des dégâts.

        Ces dernières années, l'intégration de l'IA dans les services d'incendie s'avère prometteuse.
        L'IA permet la prévision des incendies et la prise de décisions en temps réel grâce à l'analyse
        des données. Les algorithmes identifient les risques, prédisent la propagation, aidant
        les pompiers à élaborer des stratégies proactives et à allouer les ressources de manière optimale.
        L'IA fournit également des données en temps réel améliorant la connaissance de la situation.
        """
    )
    
    st.subheader("Objectif du Projet")
    st.markdown(
        """
        Ce projet vise à analyser et à visualiser les données relatives aux interventions des pompiers
        dans la ville de Londres. Nous nous concentrons sur les types d'interventions, les tendances au
        fil du temps, et les facteurs influençant les demandes d'intervention. Suite à cette analyse, l'objectif
        a été centré sur la prédiction du temps de réponse de la première brigade arrivant sur un incident à la suite d’un appel.
        """
    )

     
    st.subheader("Liens Utiles")
    st.write(
        "Pour en savoir plus sur les méthodologies utilisées et accéder aux données, veuillez consulter les liens suivants :"
    )
    
    st.markdown("[Données des Interventions](https://exemple.com/donnees-interventions)")
    st.markdown("[Documentation du Projet](https://exemple.com/documentation-projet)")

    

# Page d'exploration/Visualisation/Preprocessing
if page == pages[2] :
  

    st.title("Données et preprocessing")
    
    # Description des datasets
    st.header("Description des datasets et DataViz")
    
    st.write(
        "Nous avons deux types de datasets à notre disposition :"
    )
    
    st.markdown(
        """
        #### Incidents Records
        - **Description** : Ce dataset contient des informations sur les incidents auxquels les pompiers de Londres ont répondu.
        - **Colonnes** : 39 colonnes avec des informations variées sur les incidents.
        
        #### Mobilisation Records
        - **Description** : Ce dataset contient des informations sur le temps de réaction des véhicules de pompiers.
        - **Colonnes** : 22 colonnes avec des informations détaillées sur le temps de réaction des engins des pompiers.
        """
    )

# Affichage des data frame Incident, Mobilisation, la transformation 
    # du df Mobilisation pour le merge et le df Mergé

    
    st.subheader("  1. Dataset Mobilisation")
    st.write("""
             Dataset Mobilisation : contient les infos sur les camions déployés depuis 2009. Nottament les numéros d'incident, 
             des données sur l'identité des camions, les temps et date de mise en route, arrivée et de retour à la caserne, le lieu de déploiement,
             la raison de retard si le camion en retard

             """)
    
    st.dataframe(st.session_state["Mobilisation"].head())
    
    st.subheader("  2. Dataset Incident")
    st.write("""
             DataFrame Incident : présente les détails des incidents depuis 2009. Date, lieu d'incident, type d'incident, performance des premiers et
             second arrivant

             """)
    st.dataframe(st.session_state["Incidents"].head())
    
    
    st.subheader("  3. Préparation des données pour le merge ")
    
    # création du df qui sera mergé avec le df Incidents

    st.write("""
        Dans le dataset Mobilisation, les numéros d'incident peuvent apparaitre plusieurs fois. On ne conserve que les infos des premiers et second 
        camions arrivant. Pour n'avoir qu'une seule ligne par N° d'incident, on regroupe les variables en fonction de quel camion est concerné 
        précisant si il s'agit du 1er ou 2nd camion. Seule quelques variables sont conservées.

        """)
    
    st.dataframe(st.session_state["Mob_pompes_12"].head())
    st.caption("Dataset à merger avec le dataset Incidents")
    
    
    st.subheader("  4. Merge des Dataset")
    st.dataframe(st.session_state["Merge"].head())

    st.header("Preprocessing")
    

# Page de modélisation
if page == pages[3] :
    
    # Préparation des données pour la modélisation.
# Suppression de certaines colonnes
    @st.cache_data 
    def donnes_modelisation() :

        """""
        Préparation des données pour les modeles de reg
        """""
        
        df_2020_mod = st.session_state["df_2020"].drop(columns = ["IncidentNumber","DateOfCall","TimeOfCall","PropertyType","AddressQualifier",
                            "Postcode_full","Postcode_district","IncidentStationGround","Easting_m","Northing_m","Easting_rounded","Northing_rounded","Latitude",
                            "Longitude","Latitude_Station","Longitude_Station","NumStationsWithPumpsAttending","NumPumpsAttending","PumpCount",
                            "PumpMinutesRounded","Notional Cost (£)","NumCalls","FirstPumpArriving_TravelTimeSec",
                            "FirstPump_DelayCode_Description","FirstPump_Division_staion","tempsAPI"], axis = 1)

        df_2020_mod = df_2020_mod.dropna(axis=0)

        # Features 
        X = df_2020_mod.drop(columns=["Weekday","Month","HourOfCall","Week_Weekend","London_Zone","CalYear","Same_Incident_Station",
                                "FirstPumpArriving_AttendanceTime","AttendanceTime_Min","Periode","Periode_Rush","FirstPump_Delayed","FirstPumpArriving_TurnoutTimeSec",
                                "Station_DelayFreq","IncGeo_WardNameNew","Ward_DelayFreq","Bo_DelayFreq" ,'Incident_Fire', 'Incident_Special Service',
            'StopCode_Primary Fire', 'StopCode_Secondary Fire',"IncidentGroup",
            'StopCode_Special Service', '_Non Residential',"_Other Vehicle", '_Other Residential',
            '_Outdoor', '_Outdoor Structure', '_Road Vehicle'])

        # y pour la regression
        y_reg = df_2020_mod.FirstPumpArriving_AttendanceTime

        # y pour la classification
        y_class = df_2020_mod.AttendanceTime_Min

        # encodage de la target
        y_class = y_class.replace({'0-3min' : 0,
                '3-6min' : 1,
                '6-9min' : 2,
                "9-12min" : 3,
                '+12min' : 4
                })
        
        return X, y_reg, y_class

    X , y_reg, y_class =  donnes_modelisation()

    if "X" not in st.session_state :
        st.session_state["X"] = donnes_modelisation()[0]

    if "y_reg" not in st.session_state :
        st.session_state["y_reg"] = donnes_modelisation()[1]

    if "y_class" not in st.session_state :
        st.session_state["y_class"] = donnes_modelisation()[2]
        st.title("Modélisation")
        
    tab1, tab2, tab3 = st.tabs(["Régression", "Classification 1", "Classification 2"])
    
    with tab1 :
        
        # split pour la regression
        X_train_reg,X_test_reg,y_train_reg, y_test_reg = train_test_split(X,y_reg, test_size = 0.25, random_state=42)

        mean_enc_reg = MeanEncoder(smoothing='auto')
        X_train_reg = mean_enc_reg.fit_transform(X_train_reg,y_train_reg)
        X_test_reg = mean_enc_reg.transform(X_test_reg)

        scaler_reg = RobustScaler()
        X_train_reg_sc = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_sc = scaler_reg.transform(X_test_reg)

        # pour le modele polynomial
        @st.cache_data
        def poly_feat() :
            polynomial_features= PolynomialFeatures(degree=3)

            X_train_poly = polynomial_features.fit_transform(X_train_reg_sc)
            X_test_poly = polynomial_features.transform(X_test_reg_sc)
            return X_train_poly,X_test_poly
        
        X_train_poly,X_test_poly = poly_feat()

        #chargement des modeles
        @st.cache_resource
        def load_model_reg():
            RegressionLr = pickle.load(open('ModelesLineaire/lineaire_naif', 'rb'))
            Lasso = pickle.load(open("ModelesLineaire/Lasso", 'rb'))  
            Poly = pickle.load(open("ModelesLineaire/Poly", 'rb'))
            Elastic = pickle.load(open("ModelesLineaire/Elastic", 'rb'))
            RandomForest_reg = pickle.load(open("ModelesLineaire/RandomForest", 'rb'))
            
            return RegressionLr,Lasso,Poly,Elastic,RandomForest_reg

        RegressionLr,Lasso,Poly,Elastic,RandomForest_reg = load_model_reg()

        modeles_reg = ["Regression Linéaire", "Lasso","Polynomiale","RandomForest"
                       ,"ElasticNet"]

        dico_model_reg = { "Regression Linéaire": RegressionLr,
                   "Lasso" : Lasso,
                   "Polynomiale" : Poly,
                   "RandomForest" : RandomForest_reg,
                   "ElasticNet" : Elastic
                   }
        
        # ne pas cacher
        def calcul_metrics_reg(model):
            """""
            Calcul des metrics des modeles de régression
            """""
            # valeurs pour le set de train
            y_pred_train = model.predict(X_train_reg_sc)
            R2_train = round(r2_score(y_train_reg, y_pred_train),3)
            MAE_train = round(mean_absolute_error(y_train_reg,y_pred_train),3)
            MSE_train = round(mean_squared_error(y_train_reg,y_pred_train),3)

            # valeur pour le set de test
            y_pred_test = model.predict(X_test_reg_sc)
            R2_test = round(r2_score(y_test_reg, y_pred_test),3)
            MAE_test = round(mean_absolute_error(y_test_reg,y_pred_test),3)
            MSE_test = round(mean_squared_error(y_test_reg,y_pred_test),3)
            
            return R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test

        def calcul_metrics_poly(model):
            """""
            Calcul des metrics des modeles de régression
            """""
            # valeurs pour le set de train
            y_pred_train = model.predict(X_train_poly)
            R2_train = round(r2_score(y_train_reg, y_pred_train),3)
            MAE_train = round(mean_absolute_error(y_train_reg,y_pred_train),3)
            MSE_train = round(mean_squared_error(y_train_reg,y_pred_train),3)

            # valeur pour le set de test
            y_pred_test = model.predict(X_test_poly)
            R2_test = round(r2_score(y_test_reg, y_pred_test),3)
            MAE_test = round(mean_absolute_error(y_test_reg,y_pred_test),3)
            MSE_test = round(mean_squared_error(y_test_reg,y_pred_test),3)
            
            return R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test
        
        st.write("Modèle 1")
       
       # conserver en mémoire les valeurs pour chaque modeles

        # crée une valeur de dico model choisie contenant dico avec nom de modeles et leurs métriques
        if "model_reg_choisi" not in st.session_state :
            st.session_state["model_reg_choisi"] = {}
        
        
        # if "Reg_model_1" not in st.session_state : 
        #     st.session_state.Reg_model_1 = None

            
        reg_model = st.selectbox("Sélectionnez un modèle", modeles_reg, key = "reg_mod1")
        md_reg1 = dico_model_reg[reg_model]

        # mettre les donnée à la bonne forme pour le modele poly
        if reg_model == "Polynomiale":
            
            R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test = calcul_metrics_poly(md_reg1)
        
        else :

            R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test = calcul_metrics_reg(md_reg1)


        st.write("R2 train :", R2_train, " R2 test :", R2_test )
        st.write("MAE train :" , MAE_train," MAE test :",MAE_test)
        st.write("MSE train :", MSE_train," MSE test :" ,MSE_test)
        st.write("Différence MSE train - MSE train : " , round((MSE_train - MSE_test),3))

        
        if reg_model not in st.session_state["model_reg_choisi"] :
            st.session_state["model_reg_choisi"][reg_model] = {"R2_train" :R2_train, "MAE_train" : MAE_train, "MSE_train" :MSE_train,
                                                            "R2_test" :R2_test, "MAE_test" : MAE_test, "MSE_test" :MSE_test }



        fig = plt.figure(figsize = (15,8))
        plt.plot(md_reg1.predict(X_test_reg_sc)[0:250])
        plt.plot(y_test_reg[0:250].values)
        plt.legend(["predit","reel"])
        plt.title("Comparaison des valeurs prédites et réelles")
        st.pyplot(fig)
        

        # fonction pour les plots
        

        st.write("Modèle 2")
        @st.experimental_fragment
        def show_model_met2():
            option2 = st.selectbox("Sélectionnez un modèle", modeles_reg, key = "reg_mod2")
            md_reg2 = dico_model_reg[option2]
            
            if option2 == "Polynomiale" :
                st.write("R2 train :", calcul_metrics_poly(md_reg2)[0], " R2 test :", calcul_metrics_poly(md_reg2)[3] )
                st.write("MAE train :" , calcul_metrics_poly(md_reg2)[1]," MAE test :",calcul_metrics_poly(md_reg2)[4] )
                st.write("MSE train :", calcul_metrics_poly(md_reg2)[2]," MSE test :" ,calcul_metrics_poly(md_reg2)[5])
                st.write("Différence MSE train - MSE train : " , round((calcul_metrics_poly(md_reg2)[2] - calcul_metrics_poly(md_reg2)[5]),3))

                fig = plt.figure(figsize = (15,8))
                plt.plot(md_reg2.predict(X_test_poly)[0:250])
                plt.plot(y_test_reg[0:250].values)
                plt.legend(["predit","reel"])
                plt.title("Comparaison des valeurs prédites et réelles")
                st.pyplot(fig)

            else :
                st.write("R2 train :", calcul_metrics_reg(md_reg2)[0], " R2 test :", calcul_metrics_reg(md_reg2)[3] )
                st.write("MAE train :" , calcul_metrics_reg(md_reg2)[1]," MAE test :",calcul_metrics_reg(md_reg2)[4] )
                st.write("MSE train :", calcul_metrics_reg(md_reg2)[2]," MSE test :" ,calcul_metrics_reg(md_reg2)[5])
                st.write("Différence MSE train - MSE train : " , round((calcul_metrics_reg(md_reg2)[2] - calcul_metrics_reg(md_reg2)[5]),3))

                fig = plt.figure(figsize = (15,8))
                plt.plot(md_reg2.predict(X_test_reg_sc)[0:250])
                plt.plot(y_test_reg[0:250].values)
                plt.legend(["predit","reel"])
                plt.title("Comparaison des valeurs prédites et réelles")
                st.pyplot(fig)

        show_model_met2()

        

        import shap.explainers
        # pour visualisation des plots
        shap.plots.initjs()

        explainer = shap.explainers.LinearExplainer(Elastic, X_test_reg_sc,feature_names=X.columns)
        shap_values = explainer(X_test_reg_sc)
        
        
        # # plot des features importances
        @st.cache_data
        def sum_plot_elastic() :
            fig = plt.figure()
            shap.summary_plot(shap_values, X_test_reg_sc, plot_type="bar")
            return fig
        
        st.pyplot(sum_plot_elastic())


        # # redonne aux variables leur valeur d'origine
        X_test_reg = mean_enc_reg.inverse_transform(X_test_reg)
        shap_values.data = X_test_reg.values
        
        if "select_pred_reg" not in st.session_state :
            st.session_state["select_pred_reg"] = 0

        # permet de générer une valeur aléatoire et de faire le plot sans rerun le script à chaque
        @st.experimental_fragment
        def generate_val():
            generate_value_reg = st.button("Générer valeur aléatoire")
            if generate_value_reg :
                st.session_state["select_pred_reg"] = np.random.randint(0,len(X_test_reg))
                st.write("valeur" , st.session_state["select_pred_reg"])
            fig = plt.figure()
            shap.plots.waterfall(shap_values[st.session_state["select_pred_reg"]])
            st.pyplot(fig)

        generate_val()
        

        ### remplir des champs pour prédictions

         
        
        @st.experimental_fragment
        def generate_prediction():
                col1 , col2 = st.columns(2)

                with col1 : 
                    mois = st.number_input("Choisir un mois", min_value=0, max_value=12, step=1 , key = "month")
                    jour = st.number_input("Choisir un jour", min_value=0, max_value=7, step=1, key = "day")
                    quartier = st.selectbox("Sélectionnez un quartier", X["IncGeo_BoroughName"].unique(), key = "borought")
                    StopCode = st.selectbox("Sélectionnez l'urgence", X["StopCodeDescription"].unique(), key = "urgence")

                with col2 :        
                    heure = st.number_input("Choisir une heure", min_value=0, max_value=24, step=1, key ="hour")
                    distance = st.number_input("Choisir une distance (en km)", min_value=0.5,step=0.5 , key = "dst", format="%.2f")
                    Propriete = st.selectbox("Sélectionnez la propriété touchée", X["PropertyCategory"].unique(), key = "prop")
                    Station_deployee = st.selectbox("Sélectionnez la station déployée", X["FirstPumpArriving_DeployedFromStation"].unique(), key = "station")

                SST = st.selectbox("Sélectionnez le service special", X["SpecialServiceType"].unique(), key = "sp_serv")

                # ajouste les données temporelles aux bonnes valeurs
                heure_cos = np.cos(2*np.pi * heure/24)
                heure_sin = np.sin(2*np.pi * heure/24)
                mois_cos = np.cos(2*np.pi * mois/12)
                mois_sin = np.sin(2*np.pi * mois/12)
                jour_cos = np.cos(2*np.pi * jour/7)
                jours_sin = np.sin(2*np.pi * jour/7)


                var_prediction = pd.DataFrame(np.array([[mois_cos, mois_sin, jour_cos,jours_sin, heure_cos,heure_sin, SST,quartier,Station_deployee,distance,StopCode,Propriete ]]),
                                            columns = X_test_reg.columns)
                
                var_prediction = mean_enc_reg.transform(var_prediction)
                var_prediction_sc = scaler_reg.transform(var_prediction)

                temps = md_reg1.predict(var_prediction_sc)[0]
                
                st.write("Temsps de réponse prédit : ",temps//60,"min", round(temps%60,0),"sec")

        generate_prediction()


       


        
    # def plot_shap_summary(shap_values,X) :
    #     shap.summary_plot(shap_values, X, plot_type="bar")
    #     st.pyplot(plt.gcf())
    #     plt.clf()

    # classification bins de 3 mins
    with tab2 : 

        # Split pour la classification
        X_train_class,X_test_class,y_train_class,y_test_class = train_test_split(X,y_class, test_size = 0.25,random_state=42)

        mean_enc_class = MeanEncoder(smoothing='auto')
        X_train_class = mean_enc_class.fit_transform(X_train_class,y_train_class)
        X_test_class = mean_enc_class.transform(X_test_class)

        scaler_class = RobustScaler()
        X_train_class_sc = scaler_class.fit_transform(X_train_class)
        X_test_class_sc = scaler_class.transform(X_test_class)

        
         # chargement des modèles entrainné
        @st.cache_resource
        def load_model_class():
            RandomForest_class = pickle.load(open('ModeleClass2/RandomForest', 'rb'))
            BalancedForest = pickle.load(open("ModeleClass2/BalancedRamdomForest", 'rb'))  
            Adaboost = pickle.load(open("ModeleClass2/AdaBoostClass", 'rb'))
            DecisionTree = pickle.load(open("ModeleClass2/DecisionTree", 'rb'))
            RUSBoost = pickle.load(open("ModeleClass2/RUSBoostClassifier", 'rb'))
            
            return RandomForest_class,BalancedForest,Adaboost,DecisionTree,RUSBoost
        
        RandomForest_class,BalancedForest,Adaboost,DecisionTree,RUSBoost = load_model_class()

    # # pour arranger l'ordre des colonnes et lignes dans les rapports de class et mat de conf
        dico = {0 : "0-3min"
            ,1 : "3-6min",
            2 : "6-9min",
            3 : "9-12min",
            4 : "+12min"}
        order = ["0-3min", "3-6min", "6-9min","9-12min","+12min"]
        
        modeles = ["RandomForest", "DecisionTree","BalancedForest","Adaboost","RUSBoost"]

        # pour récupérer le model à partir de leur nom
        dico_model_class = { "RandomForest": RandomForest_class,
                   "DecisionTree" : DecisionTree,
                   "BalancedForest" : BalancedForest,
                   "Adaboost" : Adaboost,
                   "RUSBoost" : RUSBoost
                   }
        
        #fonction qui affiche le rapport de classif et la mat de conf
        def rapport_mat(model) : 
            clf = dico_model_class[model]
            y_pred_test = clf.predict(X_test_class_sc)
            y_pred_adj = [dico[i] for i in y_pred_test]
            mat = pd.crosstab(y_test_class.replace(dico),y_pred_adj,rownames=["reels"],colnames=["predits"]).reindex(index=order ,columns=order)
            rapport = pd.DataFrame(classification_report_imbalanced(y_test_class,y_pred_test, target_names = order, output_dict = True))
            
            return (mat ,rapport)

        def calcul_metrics_class(model) :
            y_pred_train = model.predict(X_train_class_sc)
            blc_acc_train = round(balanced_accuracy_score(y_train_class,y_pred_train),3)
            f1_train = round(f1_score(y_train_class,y_pred_train, average="weighted"),3)
            geo_mean_score_train = round(geometric_mean_score(y_train_class,y_pred_train,average="weighted"),3)

            y_pred_test = model.predict(X_test_class_sc)
            blc_acc_test = round(balanced_accuracy_score(y_test_class,y_pred_test),3)
            f1_test = round(f1_score(y_test_class,y_pred_test, average="weighted"),3)
            geo_mean_score_test = round(geometric_mean_score(y_test_class,y_pred_test,average="weighted"),3)
            
            return blc_acc_train,f1_train,geo_mean_score_train,blc_acc_test,f1_test,geo_mean_score_test


        # crée une valeur de dico model choisie contenant dico avec nom de modeles et leurs métriques
        if "model_class_choisi" not in st.session_state :
            st.session_state["model_class_choisi"] = {}
        
        col1,col2 = st.columns(2)

        with col1:
            
        
            st.write("Modèle 1")
            class_option1 = st.selectbox("Sélectionnez un modèle", modeles, key = "class_mod1")
            mat, rapport = rapport_mat(class_option1)
            st.write("Matrice de confusion (réels en ligne, prédits en colonne)")
            st.table(mat)
            st.write("Rapport de classification")
            st.dataframe(rapport.transpose()[0:5],width=450)
            st.write("moyennes des métriques")
            st.dataframe(rapport.iloc[0:1,5:])

            md_class_1 = dico_model_class[class_option1]
            blc_acc_train,f1_train,geo_mean_score_train,blc_acc_test,f1_test,geo_mean_score_test = calcul_metrics_class(md_class_1)
            st.write("balanced accuracy train :" , blc_acc_train)
            st.write("balanced accuracy test :" , blc_acc_test)
            st.write("F1 train :" , f1_train, "\n", " F1 test :" , f1_test)
            st.write("geometric mean train :" , geo_mean_score_train)
            st.write(" geometic mean test :" , geo_mean_score_test)


            if class_option1 not in st.session_state["model_class_choisi"] :
                st.session_state["model_class_choisi"][class_option1] = {"balanced_accuracy_train" :blc_acc_train, "F1_train" : f1_train, "geometric_mean_train" :geo_mean_score_train,
                                                                "balanced_accuracy_test" :blc_acc_test, "F1_test" : f1_test, "geometric_mean_test" : geo_mean_score_test }


        with col2 :
            st.write("Modèle 2")
            class_option2 = st.selectbox("Sélectionnez un modèle", modeles,key = "class_mod2 ")

            
                
            mat, rapport = rapport_mat(class_option2)
            st.write("Matrice de confusion (réels en ligne, prédits en colonne)")
            st.table(mat)
            st.write("Rapport de classification")
            st.dataframe(rapport.transpose()[0:5],width=450)
            st.write("moyennes des métriques")
            st.dataframe(rapport.iloc[0:1,5:])

            
            @st.experimental_fragment
            def show_model_met2() :
                md_class_2 = dico_model_class[class_option2]
                blc_acc_train, f1_train, geo_mean_score_train, blc_acc_test,f1_test,geo_mean_score_test = calcul_metrics_class(md_class_2)
                st.write("balanced accuracy train :" , blc_acc_train)
                st.write("balanced accuracy test :" , blc_acc_test)
                st.write("F1 train :" , f1_train,  "F1 test :" , f1_test)
                st.write("geometric mean train :" , geo_mean_score_train)
                st.write( "geometic mean test :" , geo_mean_score_test)

                show_model_met2()
            
            

        @st.experimental_fragment
        def generate_prediction():
            model_class_pred = st.selectbox("Sélectionnez un modèle", modeles,key = "mod_class_pred")
            prediction_md = dico_model_class[model_class_pred]
            
            col1 , col2 = st.columns(2)

            with col1 : 
                mois = st.number_input("Choisir un mois", min_value=0, max_value=12, step=1 , key = "class_month")
                jour = st.number_input("Choisir un jour", min_value=0, max_value=7, step=1, key = "class_day")
                quartier = st.selectbox("Sélectionnez un quartier", X["IncGeo_BoroughName"].unique(), key = "class_borought")
                StopCode = st.selectbox("Sélectionnez l'urgence", X["StopCodeDescription"].unique(), key = "class_urgence")

            with col2 :        
                heure = st.number_input("Choisir une heure", min_value=0, max_value=23, step=1, key ="class_hour")
                distance = st.number_input("Choisir une distance (en km)", min_value=0.5,step=0.5 , key = "class_dst",format = "%.2f")
                Propriete = st.selectbox("Sélectionnez la propriété touchée", X["PropertyCategory"].unique(), key = "class_prop")
                Station_deployee = st.selectbox("Sélectionnez la station déployée", X["FirstPumpArriving_DeployedFromStation"].unique(), key = "class_station")

            SST = st.selectbox("Sélectionnez le service special", X["SpecialServiceType"].unique(), key = "class_sp_serv")

            # ajouste les données temporelles aux bonnes valeurs
            heure_cos = np.cos(2*np.pi * heure/24)
            heure_sin = np.sin(2*np.pi * heure/24)
            mois_cos = np.cos(2*np.pi * mois/12)
            mois_sin = np.sin(2*np.pi * mois/12)
            jour_cos = np.cos(2*np.pi * jour/7)
            jours_sin = np.sin(2*np.pi * jour/7)


            var_prediction = pd.DataFrame(np.array([[mois_cos, mois_sin, jour_cos,jours_sin, heure_cos,heure_sin, SST,quartier,Station_deployee,distance,StopCode,Propriete ]]),
                                        columns = X_test_reg.columns)
            
            var_prediction = mean_enc_reg.transform(var_prediction)
            var_prediction_sc = scaler_reg.transform(var_prediction)

            pred = prediction_md.predict(var_prediction_sc)[0]
            temps =dico[pred]
            # prends l'argument du dico pour renvoiyer la classe
            # dico = {0 : "0-3min"
            # ,1 : "3-6min",
            # 2 : "6-9min",
            # 3 : "9-12min",
            # 4 : "+12min"}

            st.write("Temsps de réponse prédit :" , temps)

        generate_prediction()

        A = st.checkbox("", key = "plot1")
        if A : 
            order = ["0-3min","3-6min","6-9min","9-12min","+12min"]
            @st.cache_data
            def plot_dst() :
                fig = plt.figure(figsize = (15,8))
                sns.boxplot(x = st.session_state["df_2020"].AttendanceTime_Min,y = st.session_state["df_2020"].dst_StationIncident ,data = st.session_state["df_2020"],notch = True, order = order)
                plt.title("Distance Incident-Station en fonction \n de la perforamnce de réponse du 1er camion");
                return fig
            st.pyplot(plot_dst())
        
        B = st.checkbox("",key = "plot2")
        if B : 
            @st.cache_data
            def plot_dst_hue() :
                fig = plt.figure(figsize = (15,8))
                sns.boxplot(x =st.session_state["df_2020"].AttendanceTime_Min,y = st.session_state["df_2020"].dst_StationIncident ,hue = st.session_state["df_2020"].FirstPump_Delayed,data = st.session_state["df_2020"],notch = True, order = order)
                plt.title("Distance Incident-Station en fonction \n de la perforamnce de réponse du 1er camion");
                return fig
            st.pyplot(plot_dst_hue())    

        ## shap 
       

    

    # classification bins de diqtribution égales
    with tab3 : 
        st.write("2e classifcation")
        
        @st.cache_data
        def load_preprocess_data():
            #load du df
            df = pd.read_csv('result_filtered.csv')

            df['DateTimeOfCall'] = pd.to_datetime(df['DateTimeOfCall'])
            df['hour'] = df['DateTimeOfCall'].dt.hour
            df['day_of_week'] = df['DateTimeOfCall'].dt.dayofweek
            df['day_of_month'] = df['DateTimeOfCall'].dt.day
            df['month'] = df['DateTimeOfCall'].dt.month
            df['day_of_year'] = df['DateTimeOfCall'].dt.dayofyear
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            df.drop(['hour', 'day_of_week', 'day_of_month', 'month', 'day_of_year', 'DateTimeOfCall', 
                    'Latitude', 'Longitude', 'latitude_deployed_station', 'longitude_deployed_station'], axis=1, inplace=True)

            # Binning et encodage
            bins = [1.0, 211.0, 269.0, 325.0, 405.0, 1200.0]
            class_labels = ['0', '1', '2', '3', '4']

            # Assignation des classes en utilisant pd.cut
            df['Class'] = pd.cut(df['FirstPumpArriving_AttendanceTime'], bins=bins, labels=class_labels, include_lowest=True)

            # Ordinal Encoding
            df['Class_Encoded'] = df['Class'].astype(int)  # Convertir les labels en entiers

            # Supprimer les colonnes inutiles
            df.drop(['Class', 'FirstPumpArriving_AttendanceTime'], axis=1, inplace=True)
            
            # Prépare les données pour modeles
            X = df.drop(columns=['Class_Encoded'])
            y = df['Class_Encoded']
            
            return X , y

        X, y = load_preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        train_data = X_train.copy()
        train_data['Class_Encoded'] = y_train.values

        categorical_cols = ['IncidentGroup', 'PropertyCategory', 'IncGeo_BoroughName', 'IncGeo_WardName',
                            'IncidentStationGround', 'FirstPumpArriving_DeployedFromStation', 'Nom_station']

        for col in categorical_cols:
            mean_encoded_col = train_data.groupby(col)['Class_Encoded'].mean()
            global_mean = y_train.mean()
            X_train[col + '_mean_encoded'] = X_train[col].map(mean_encoded_col).fillna(global_mean)
            X_test[col + '_mean_encoded'] = X_test[col].map(mean_encoded_col).fillna(global_mean)

        X_train.drop(columns=categorical_cols, inplace=True)
        X_test.drop(columns=categorical_cols, inplace=True)

        scaler = StandardScaler()
        numerical_cols = X_train.columns
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        modeles_charge = {
            "LogisticRegression": pickle.load(open('ModelesClassW/Bin5/logistic_regression_model.pkl', 'rb')),
            "TensorflowKeras": pickle.load(open('ModelesClassW/Bin5/TensorflowKeras.pkl', 'rb')),
            "XGBoost": pickle.load(open('ModelesClassW/Bin5/XGBoost.pkl', 'rb')),
            }
            
            # pour arranger l'ordre des colonnes et lignes dans les rapports de class et mat de conf
        dico = {0 : "1.0s - 211.0s",
                    1 : "211.0s - 269.0s",
                    2 : "269.0s - 325.0s",
                    3 : "325.0s - 405.0s",
                    4 : "405.0s - 1200.0s"}
        
        order = ["1.0s - 211.0s", "211.0s - 269.0s", "269.0s - 325.0s","325.0s - 405.0s","405.0s - 1200.0s"]
            
            # pour récupérer le model à partir de leur nom
        dico_model = {  "LogisticRegression": modeles_charge.get("LogisticRegression"),
                            "TensorflowKeras": modeles_charge.get("TensorflowKeras"),
                            "XGBoost": modeles_charge.get("XGBoost"),
                        }
        # Préparer les données pour les modèles
        
        # X_train, X_test, y_train, y_test = load_preprocess_data()
        
        def rapport_mat(model_name):
                model = dico_model.get(model_name)
                if model is None:
                    st.error(f"Modèle {model_name} non trouvé.")
                    return None, None
        
                if model_name == "TensorflowKeras":
                    y_pred_prob = model.predict(X_test)
                    y_pred = np.argmax(y_pred_prob, axis=1) #+ 1  # Assurez-vous que les classes commencent à 1
                
                elif model_name == "XGBoost":
                    dmatrix_test = xgb.DMatrix(data =X_test, label = y_test)
                    # matrice de proba de prédiction
                    pred = model.predict(dmatrix_test, output_margin = True)
                    # initialise l'array/liste des résulats
                    y_pred = []
                    # pour chaque ligne, on récupère la position de la proba max == correspond à la classe
                    for row in range(pred.shape[0]) :
                        classe = np.argmax(pred[row])
                        y_pred.append(classe)

                else:
                    y_pred = model.predict(X_test)

                mat = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
                rapport = classification_report(y_test, y_pred, target_names=order, output_dict=True)
                return mat, pd.DataFrame(rapport).transpose()
            
                # Liste des modèles pour la sélection
        
        modeles = ["LogisticRegression", "RandomForestClassifier", "TensorflowKeras", "XGBoost"]

            # Sélection du modèle 1
        
        
        st.write("Modèle 1")
        class_bin = st.selectbox("Sélectionnez un modèle", modeles, key= "modelBin5")
        st.write("selection", class_bin)

        mat, rapport = rapport_mat(class_bin)
         # Affichage des résultats pour le modèle 1
        st.table(mat)
        st.dataframe(rapport)
        
        #  Sélection du modèle 2
        st.write("Modèle 2")
        class_bin2 = st.selectbox("Sélectionnez un modèle", modeles, key=6)
        mat, rapport = rapport_mat(class_bin2)
        
        # Affichage des résultats pour le modèle 2
        st.table(mat)
        st.dataframe(rapport)


# Page de conclusion
def conclusion_page() :
    st.write("page de conclusion/ouverture")


