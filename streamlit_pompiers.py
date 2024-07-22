import streamlit as st
import base64

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
    st.title("Exploration")
    st.write("Ceci est la page d'exploration.")
    st.write("Vous pouvez explorer les données ici.")

# Page de visualisation des données
def dataviz_page():
    st.title("DataVizualization")
    st.write("Ceci est la page de visualisation des données.")
    st.write("Vous pouvez voir les visualisations ici.")

# Page de modélisation
def modeling_page():
    st.title("Modélisation")
    st.write("Ceci est la page de modélisation.")
    st.write("Vous pouvez modéliser les données ici.")

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
