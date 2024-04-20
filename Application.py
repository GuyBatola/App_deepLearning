import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Liste des modèles disponibles
MODELS = ['Arbre de décision', 'Forêt aléatoire', 'XGBoost', 'Réseaux de neurones', 'Réseau de neurones récurrents']

# Chargement du vocabulaire et des modèles pré-entrainés
vectorize = pickle.load(open("vectorizer.pkl", "rb"))
    
tree = pickle.load(open("tree.pkl", "rb"))
xgb = pickle.load(open("xgb.pkl", "rb"))
neural = pickle.load(open("neural.pkl", "rb"))
# Sélection du modèle
selected_model = st.sidebar.selectbox("Sélectionnez un modèle", MODELS)

# Affichage de l'explication du modèle sélectionné
if selected_model == 'Arbre de décision':
    st.sidebar.markdown("""
    L'arbre de décision est un modèle d'apprentissage supervisé utilisé pour la classification et la régression. Il divise l'espace des caractéristiques en partitions rectangulaires, chaque partition étant associée à une classe ou une valeur de sortie.
    """)
    model = tree
elif selected_model == 'Forêt aléatoire':
    st.sidebar.markdown("""
    La forêt aléatoire est un ensemble d'arbres de décision. Chaque arbre est construit sur un sous-ensemble aléatoire des données d'entraînement et utilise une sous-ensemble aléatoire des caractéristiques pour la division.
    """)
    model = xgb
elif selected_model == 'XGBoost':
    st.sidebar.markdown("""
    XGBoost est une implémentation optimisée du gradient boosting. Il combine plusieurs modèles faibles pour former un modèle fort en utilisant le gradient de la fonction de perte.
    """)
    model = xgb
elif selected_model == 'Réseau de neurones':
    st.sidebar.markdown("""
    Le réseau de neurones avec Word Embedding utilise une couche d'embedding pour convertir les mots en vecteurs denses avant de les passer au réseau de neurones. Cela permet au modèle de capturer les relations sémantiques entre les mots.
    """)
    model = neural
elif selected_model == 'Réseau de neurones récurrents':
    st.sidebar.markdown("""
    Le réseau de neurones récurrents (RNN) est un type de réseau de neurones capable de traiter des données séquentielles. Il utilise des boucles récurrentes pour partager les mêmes poids à chaque pas de temps, ce qui permet de capturer les dépendances séquentielles dans les données.
    """)
    model = neural

st.write('# Projet Deep Learning')
st.write('## Auteurs : Guy BATOLA, Yann OYE, Idrissa Belem, Alimatou DIOP')
st.write('### Objectif : ')
st.write("""construction un modèle capable de prédire le sentiment sur "Fine Foods" et de le déployer sous forme de service web pour votre client.""")

# Zone de texte pour saisir le commentaire
comment = st.text_area("Saisissez votre commentaire ici")

# Bouton pour soumettre le commentaire
if st.button("Valider"):
    if comment:
        # tree
        if selected_model == 'tree':
            a = vectorize.transform([comment])
            sentiment = model.predict(a)
            st.write("Sentiment prédit : ", sentiment)
        # RF et XGB
        elif selected_model == 'Forêt aléatoire' or selected_model == 'XGBoost' :
            a = vectorize.transform([comment])
            sentiment = model.predict(a)
            if sentiment==2 :
                sentiment2 = "Positif"
                st.write("Sentiment prédit : ", sentiment2)
            elif sentiment==1 :
                sentiment2 = "Negatif"
                st.write("Sentiment prédit : ", sentiment2)
            else :
                sentiment2 = "Mitige"
                st.write("Sentiment prédit : ", sentiment2)
        else:
            a = vectorize.transform([comment])
            sentiment = model.predict(a.toarray()).argmax(-1)
            if sentiment==2 :
                sentiment2 = "Positif"
                st.write("Sentiment prédit : ", sentiment2)
            elif sentiment==1 :
                sentiment2 = "Negatif"
                st.write("Sentiment prédit : ", sentiment2)
            else :
                sentiment2 = "Mitige"
                st.write("Sentiment prédit : ", sentiment2)
    else:
        st.warning("Veuillez saisir un commentaire.")
