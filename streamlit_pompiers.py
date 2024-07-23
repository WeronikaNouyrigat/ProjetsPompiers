import streamlit as st
import base64

from sklearn.model_selection import train_test_split
from feature_engine.encoding import MeanEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error, root_mean_squared_error,r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from imblearn.metrics import macro_averaged_mean_absolute_error ,classification_report_imbalanced, geometric_mean_score , sensitivity_score

import shap
import pickle

# Fonction pour lire l'image et la convertir en base64
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
@st.cache_data  
def Data_Mob():
    dico = {'IncidentNumber' : 'str',
         'DelayCodeId' : 'object',
         "CalYear" : 'str',
        "ResourceMobilisationId" : "str"}
    Mobilisation = pd.read_csv("Data/Mobilisation_2009_2023.csv", dtype=dico)
    Mobilisation = Mobilisation.drop(["TempsDepartSeconds", "TempsTrajetSeconds"], axis = 1)

    return Mobilisation

# fonction pour charger le df Incident
@st.cache_data  
def Data_Incident():
    dico = {'IncidentNumber' : 'str',
                "CalYear" : 'str'}
    Incidents = pd.read_csv("Data/Incidents_2009_2024.csv", dtype = dico)
    return Incidents

# fonction pour charger le df mergé
@st.cache_data  
def Data_Merge():
    dico = {'IncidentNumber' : 'str',
             "CalYear" : "str",
                 "Easting_m" : 'str',
                 "Northing_m" : "str",
                 "UPRN" : "str" ,
                 "USRN" : "str",
                 "Easting_rounded" : "str",
                 "Northing_rounded" : "str"
         }
    Merge = pd.read_csv("Data/Data_Mergees.csv", dtype = dico)
    return Merge

# le dataFrame avec les données encodées et filtre 2020
@st.cache_data
def Data_Mod():
    dico = {'IncidentNumber' : 'str',     
         }
    df = pd.read_csv("Data/Data_Encodee_V2_2_2020.csv", dtype = dico)
    # ncr de rename les colonnes avec les noms d'origine car les nouvelles étaient utilisées pour conserver les  valeurs originales après un pd.getdummies
    df = df.rename(columns={"IncidentGroup_orig" : "IncidentGroup",
                       "StopCodeDescription_orig" : "StopCodeDescription",
                       "PropertyCategory_orig" : "PropertyCategory"})

    df["dst_StationIncident"] = df["dst_StationIncident"]/1000

    return df

# chargement des datasets
Mobilisation = Data_Mob()
Incidents = Data_Incident()
Merge = Data_Merge()
df_2020 = Data_Mod()

# Titre de la page
def page_title():
    st.markdown('<div class="container"><h1 class="title">Projet Pompiers de la ville de Londres</h1></div>', unsafe_allow_html=True)

# Page d'accueil
def home_page():
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
    page_title()
    st.sidebar.title("Navigation")
    st.sidebar.write("Bienvenue sur l'application de gestion des pompiers.")
    st.sidebar.write("Veuillez sélectionner une option dans le menu.")

# Page d'introduction
def introduction_page():
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



# Page d'exploration
def exploration_page():
    st.title("Exploration des données")
    
    # Description des datasets
    st.subheader("Description des datasets")
    
    st.write(
        "Nous avons deux types de datasets à notre disposition :"
    )
    
    st.markdown(
        """
        ### 1. Incidents Records
        - **Description** : Ce dataset contient des informations sur les incidents auxquels les pompiers de Londres ont répondu.
        - **Colonnes** : 39 colonnes avec des informations variées sur les incidents.
        
        ### 2. Mobilisation Records
        - **Description** : Ce dataset contient des informations sur le temps de réaction des véhicules de pompiers.
        - **Colonnes** : 22 colonnes avec des informations détaillées sur le temps de réaction des engins des pompiers.
        """
    )

# Affichage des data frame Incident, Mobilisation, la transformation 
    # du df Mobilisation pour le merge et le df Mergé
    st.title("Présentation des données")
    
    st.subheader("1. Dataset Mobilisation")
    st.write("""
             Dataset Mobilisation : contient les infos sur les camions déployés depuis 2009. Nottament les numéros d'incident, 
             des données sur l'identité des camions, les temps et date de mise en route, arrivée et de retour à la caserne, le lieu de déploiement,
             la raison de retard si le camion en retard

             """)
    
    st.dataframe(Mobilisation.head())
    
    st.subheader("2. Dataset Incident")
    st.write("""
             DataFrame Incident : présente les détails des incidents depuis 2009. Date, lieu d'incident, type d'incident, performance des premiers et
             second arrivant

             """)
    st.dataframe(Incidents.head())
    
    
    st.subheader("3. Préparation des données pour le merge ")
    
    # création du df qui sera mergé avec le df Incidents
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
    st.write("""
             Dans le dataset Mobilisation, les numéros d'incident peuvent apparaitre plusieurs fois. On ne conserve que les infos des premiers et second 
             camions arrivant. Pour n'avoir qu'une seule ligne par N° d'incident, on regroupe les variables en fonction de quel camion est concerné 
             précisant si il s'agit du 1er ou 2nd camion. Seule quelques variables sont conservées.

             """)
    
    st.dataframe(Mob_pompes_12.head())
    st.caption("Dataset à merger avec le dataset Incidents")
    
    
    st.subheader("4. Merge des Dataset")
    st.dataframe(Merge.head())
    
    


# Page de visualisation des données
def dataviz_page():
    st.title("DataVizualization")
    st.write("Ceci est la page de visualisation des données.")
    st.write("Vous pouvez voir les visualisations ici.")


    fig = plt.figure(figsize = (10,7))
    plt.subplot(2,1,1)
    plt.bar(x=(Merge.FirstPumpArriving_AttendanceTime//60).value_counts().index,
            height=(Merge.FirstPumpArriving_AttendanceTime//60).value_counts(),
            label = "premier camion")
    plt.xticks(ticks =range(0,21))
    plt.legend()
    plt.title("Distribution du temps de réaction du premier camion sur place en minutes")
    plt.annotate(f'Moyenne:{"%02d:%02d" % (divmod(Merge.FirstPumpArriving_AttendanceTime.mean(),60))} \nÉcart-type :{"%02d:%02d" % (divmod(Merge.FirstPumpArriving_AttendanceTime.std(),60))} \nMédiane :{"%02d:%02d" % (divmod(Merge.FirstPumpArriving_AttendanceTime.median(),60))}',xy=(17,210000))

    plt.subplot(2,1,2)
    plt.bar(x = (Merge.SecondPumpArriving_AttendanceTime//60).value_counts().index,
            height = (Merge.SecondPumpArriving_AttendanceTime//60).value_counts(),
            color ='red', label = "second camion")
    plt.xticks(ticks = range(0,21))
    plt.xlabel("Temps de réponse en minutes")
    plt.title("Distribution du temps de réaction du second camion sur place en minutes")
    plt.annotate(f'Moyenne:{"%02d:%02d" % (divmod(Merge.SecondPumpArriving_AttendanceTime.mean(),60))}\nÉcart-type :{"%02d:%02d" % (divmod(Merge.FirstPumpArriving_AttendanceTime.std(),60))} \nMédiane :{"%02d:%02d" % (divmod(Merge.FirstPumpArriving_AttendanceTime.median(),60))}',xy=(17,70000))

    plt.legend()
    st.pyplot(fig)

    # plot de temps de réaction en fonction inner outer
    zone = df_2020.groupby("London_Zone")[["FirstPumpArriving_AttendanceTime"]].agg(["mean","std","median"])

    inner_mean,inner_std,inner_median = tuple(zone.loc[1,"FirstPumpArriving_AttendanceTime"].map(lambda x : "%02d:%02d" % divmod(x,60)))
    outer_mean,outer_std,outer_median = tuple(zone.loc[0,"FirstPumpArriving_AttendanceTime"].map(lambda x : "%02d:%02d" % divmod(x,60)))

    fig,ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data =df_2020, y = df_2020.FirstPumpArriving_AttendanceTime//60,x = df_2020.London_Zone,
                errorbar = 'ci')

    plt.xlabel("Zone d'arrondissement")
    plt.ylabel("Temps de réaction (minutes)")
    plt.title("Temps de réaction moyen des 1er camions arrivés en fonction de la zone de Londres")

    ax.bar_label(ax.containers[0],labels = [f'Moyenne:{outer_mean}\nÉcart-type :{outer_std} \nMédiane :{outer_median}',f'Moyenne:{inner_mean}\nÉcart-type :{inner_std} \nMédiane :{inner_median}'] ,fmt='Mean:')
    plt.xticks([0,1],["Outer London", "Inner London"])

    plt.ylim(0,7)
    st.pyplot(fig)

    box = st.checkbox("Résumé des stats")
    if box : 
        st.table(df_2020.groupby('London_Zone')["FirstPumpArriving_AttendanceTime"].agg(['mean','var','std']))

    # temps de réponse en fonction de l'arrondissement
    fig = plt.figure(figsize = (20,15))
    sns.boxenplot(y = df_2020["FirstPumpArriving_AttendanceTime"] , x= "IncGeo_BoroughName"
                ,data = df_2020, hue = "London_Zone")
    plt.xticks(rotation=90)
    plt.yticks(ticks = range(0,1260,60),labels= range(0,21))
    plt.legend(loc = "upper right")
    plt.title(" Temps de réponse 1er camion en fonction du quartier")
    st.pyplot(fig)
        
    # fort impacte de si il y a retard ou non sur le temps d'arrivé
    fig = plt.figure()
    sns.boxplot(y =df_2020.FirstPumpArriving_AttendanceTime,x = df_2020.FirstPump_Delayed)
    plt.yticks(range(0,1300,120), labels=list(range(0,22,2)))
    plt.title("temps de réponse en fonction de si il y a retard")
    st.pyplot(fig)


    # faire un truc de selection pour les variables catégorielles avec des barplots
    # type de propriete, type d'urgence, temporelle
    st.write("plot de variables catégorielles en fonction du temps de réponse en selection quelles variable on veut à parti d'une selesctbox")
    #plt.figure(figsize = (7,5))
    #sns.barplot(x ="FirstPumpArriving_AttendanceTime", y = df_2020.PropertyCategory, data =df_2020)
    #plt.title("Temps de réponse moyen en fonction de proporiete touchée")
    #st.pyplot(fig)




# Préparation des données pour la modélisation.
# Suppression de certaines colonnes
df_2020_mod = df_2020.drop(columns = ["IncidentNumber","DateOfCall","TimeOfCall","PropertyType","AddressQualifier",
                    "Postcode_full","Postcode_district","IncidentStationGround","Easting_m","Northing_m","Easting_rounded","Northing_rounded","Latitude",
                    "Longitude","Latitude_Station","Longitude_Station","NumStationsWithPumpsAttending","NumPumpsAttending","PumpCount",
                    "PumpMinutesRounded","Notional Cost (£)","NumCalls","FirstPumpArriving_TravelTimeSec",
                    "FirstPump_DelayCode_Description","FirstPump_Division_staion","tempsAPI"], axis = 1)

df_2020_mod = df_2020_mod.dropna(axis=0)

# Features 
X = df_2020_mod.drop(columns=["Weekday","Month","HourOfCall","Week_Weekend","London_Zone","CalYear","Same_Incident_Station",
                          "FirstPumpArriving_AttendanceTime","AttendanceTime_Min","Periode","Periode_Rush","FirstPump_Delayed","FirstPumpArriving_TurnoutTimeSec",
                          "Station_DelayFreq","IncGeo_WardNameNew","Ward_DelayFreq","Bo_DelayFreq" ,'Incident_Fire', 'Incident_Special Service',
       'StopCode_Primary Fire', 'StopCode_Secondary Fire',
       'StopCode_Special Service', '_Non Residential', '_Other Residential',
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

# Séparation train test et encodage, standardisation 
#pour la regression
X_train_reg,X_test_reg,y_train_reg,y_test_reg = train_test_split(X,y_reg, test_size = 0.25)

mean_enc = MeanEncoder(smoothing='auto',unseen='encode')
X_train_reg = mean_enc.fit_transform(X_train_reg,y_train_reg)
X_test_reg = mean_enc.transform(X_test_reg)

scaler = RobustScaler()
X_train_reg_sc = scaler.fit_transform(X_train_reg)
X_test_reg_sc = scaler.transform(X_test_reg)

# pour la classification
X_train_class,X_test_class,y_train_class,y_test_class = train_test_split(X,y_class, test_size = 0.25)

mean_enc = MeanEncoder(smoothing='auto',unseen='encode')
X_train_class = mean_enc.fit_transform(X_train_class,y_train_class)
X_test_class = mean_enc.transform(X_test_class)

scaler = RobustScaler()
X_train_class_sc = scaler.fit_transform(X_train_class)
X_test_class_sc = scaler.transform(X_test_class)



# Page de modélisation
def modeling_page():
    st.title("Modélisation")
    tab1, tab2, tab3 = st.tabs(["Régression", "Classification 1", "Classification 2"])
    
    with tab1 :
        #chargement des modeles
        RegressionLr = pickle.load(open('ModelesLineaire/lineaire_naif', 'rb'))
        Lasso = pickle.load(open("ModelesLineaire/Lasso", 'rb'))  
        Poly = pickle.load(open("ModelesLineaire/Poly", 'rb'))
        Elastic = pickle.load(open("ModelesLineaire/Elastic", 'rb'))
        RandomForest_reg = pickle.load(open("ModelesLineaire/RandomForest", 'rb'))

        modeles_reg = ["Regression Linéaire", "Lasso","Polynomiale","RandomForest"
                       ,"ElasticNet"]

        dico_model_reg = { "Regression Linéaire": RegressionLr,
                   "Lasso" : Lasso,
                   "Polynomiale" : Poly,
                   "RandomForest" : RandomForest_reg,
                   "ElasticNet" : Elastic
                   }
        
        def calcul_metrics_reg(model):
            y_pred = model.predict(X_test_reg_sc)
            R2 = model.score(y_test_reg, y_pred)
            MAE = mean_absolute_error(y_test_reg,y_pred)
            MSE = mean_squared_error(y_test_reg,y_pred)
            
            return R2, MAE, MSE

        st.write("Modèle 1")
        option1 = st.selectbox("Sélectionnez un modèle", modeles_reg, key = 1)
        md = dico_model_reg[option1]
        # st.write("R2 :", calcul_metrics_reg(md)[0])
        # st.write("MAE :" , calcul_metrics_reg(md)[1])
        # st.write("MSE :", calcul_metrics_reg(md)[2])
        
        st.write("Rajout de bar pour choisir la range de values à prédir")
        
        fig = plt.figure(figsize = (15,8))
        plt.plot(md.predict(X_test_reg_sc)[0:250])
        plt.plot(y_test_reg[0:250].values)
        plt.legend(["predit","reel"])
        plt.title("Comparaison des valeurs prédites et réelles")
        st.pyplot(fig)

        st.write("Modèle 2")
        option2 = st.selectbox("Sélectionnez un modèle", modeles_reg, key = 2)
        md = dico_model_reg[option2]
        st.write("R2 :", calcul_metrics_reg(md)[0])
        st.write("MAE :" , calcul_metrics_reg(md)[1])
        st.write("MSE :", calcul_metrics_reg(md)[2])
        
        fig = plt.figure(figsize = (15,8))
        plt.plot(md.predict(X_test_reg_sc)[0:250])
        plt.plot(y_test_reg[0:250].values)
        plt.legend(["predit","reel"])
        plt.title("Comparaison des valeurs prédites et réelles")
        st.pyplot(fig)

        st.write("ajout des plot shap et possiblitées de faire des prédictions en choisissant les valeurs des variables?")

        import shap.explainers
        # pour visualisation des plots
        shap.plots.initjs()

        explainer = shap.explainers.LinearExplainer(Elastic, X_test_reg_sc,feature_names=X.columns)
        shap_values = explainer(X_test_reg_sc)
        # plot des features importances
        fig = plt.figure()
        shap.summary_plot(shap_values, X_test_reg_sc, plot_type="bar")
        st.pyplot(fig)

        # redonne aux variables leur valeur d'origine
        shap_values.data = X_test.values
        
        fig = plt.figure()
        shap.plots.waterfall(shap_values[678])
        st.pyplot(fig)


def plot_shap_summary(shap_values,X) :
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(plt.gcf())
    plt.clf()


    with tab2 : 
        st.write("tab2")
        # chargement des modèles entrainné
    #     RandomForest_class = pickle.load(open('ModeleClass2/RandomForest', 'rb'))
    #     BalancedForest = pickle.load(open("ModeleClass2/BalancedRamdomForest", 'rb'))  
    #     Adaboost = pickle.load(open("ModeleClass2/AdaBoostClass", 'rb'))
    #     DecisionTree = pickle.load(open("ModeleClass2/DecisionTree", 'rb'))
    #     RUSBoost = pickle.load(open("ModeleClass2/RUSBoostClassifier", 'rb'))

    # # pour arranger l'ordre des colonnes et lignes dans les rapports de class et mat de conf
    #     dico = {0 : "0-3min"
    #         ,1 : "3-6min",
    #         2 : "6-9min",
    #         3 : "9-12min",
    #         4 : "+12min"}
    #     order = ["0-3min", "3-6min", "6-9min","9-12min","+12min"]
        
    #     # pour récupérer le model à partir de leur nom
    #     dico_model = { "RandomForest": RandomForest_class,
    #                "DecisionTree" : DecisionTree,
    #                "BalancedForest" : BalancedForest,
    #                "Adaboost" : Adaboost,
    #                "RUSBoost" : RUSBoost
    #                }
        
    #     #fonction qui affiche le rapport de classif et la mat de conf
    #     def rapport_mat(model) : 
    #         clf = dico_model[model]
    #         y_pred_test = clf.predict(X_test_class_sc)
    #         y_pred_adj = [dico[i] for i in y_pred_test]
    #         mat = pd.crosstab(y_test_class.replace(dico),y_pred_adj,rownames=["reels"],colnames=["predits"]).reindex(index=order ,columns=order)
    #         rapport = pd.DataFrame(classification_report_imbalanced(y_test_class,y_pred_test, target_names = order, output_dict = True))
            
    #         return (mat ,rapport)

    #     modeles = ["RandomForest", "DecisionTree","BalancedForest","Adaboost","RUSBoost"]

    #     st.write("Modèle 1")
    #     option1 = st.selectbox("Sélectionnez un modèle", modeles, key = 1)
    #     mat, rapport = rapport_mat(option1)
    #     st.table(mat,width=1000)
    #     st.dataframe(rapport.transpose()[0:5],width=450)
    #     st.dataframe(rapport.iloc[0:1,5:])

    #     st.write("Modèle 2")
    #     option2 = st.selectbox("Sélectionnez un modèle", modeles,key = 2)
    #     mat, rapport = rapport_mat(option2)
    #     st.dataframe(mat)
    #     st.dataframe(rapport.transpose()[0:5])
    #     st.dataframe(rapport.iloc[0:1,5:])

    with tab3 : 
        st.write("2e classifcation")


# Page principale
def main():
    pages = {
        "Home": home_page,
        "Introduction": introduction_page,
        "Exploration": exploration_page,
        "DataVizualization": dataviz_page,
        "Modélisation": modeling_page
    }
    
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Aller vers", list(pages.keys()))
    
    # Appeler la fonction correspondant à la page sélectionnée
    pages[page_selection]()

if __name__ == "__main__":
    main()
