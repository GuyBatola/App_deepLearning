import streamlit as st

# Liste des modèles disponibles
MODELS = ['Arbre de décision', 'Forêt aléatoire', 'XGBoost', 'Réseaux de neurones avec Word Embedding', 'Réseau de neurones récurrents']

# Sélection du modèle
selected_model = st.sidebar.selectbox("Sélectionnez un modèle", MODELS)

# Affichage de l'explication du modèle sélectionné
if selected_model == 'Arbre de décision':
    st.sidebar.markdown("""
    L'arbre de décision est un modèle d'apprentissage supervisé utilisé pour la classification et la régression. Il divise l'espace des caractéristiques en partitions rectangulaires, chaque partition étant associée à une classe ou une valeur de sortie.
    """)
elif selected_model == 'Forêt aléatoire':
    st.sidebar.markdown("""
    La forêt aléatoire est un ensemble d'arbres de décision. Chaque arbre est construit sur un sous-ensemble aléatoire des données d'entraînement et utilise une sous-ensemble aléatoire des caractéristiques pour la division.
    """)
elif selected_model == 'XGBoost':
    st.sidebar.markdown("""
    XGBoost est une implémentation optimisée du gradient boosting. Il combine plusieurs modèles faibles pour former un modèle fort en utilisant le gradient de la fonction de perte.
    """)
elif selected_model == 'Réseau de neurones avec Word Embedding':
    st.sidebar.markdown("""
    Le réseau de neurones avec Word Embedding utilise une couche d'embedding pour convertir les mots en vecteurs denses avant de les passer au réseau de neurones. Cela permet au modèle de capturer les relations sémantiques entre les mots.
    """)
elif selected_model == 'Réseau de neurones récurrents':
    st.sidebar.markdown("""
    Le réseau de neurones récurrents (RNN) est un type de réseau de neurones capable de traiter des données séquentielles. Il utilise des boucles récurrentes pour partager les mêmes poids à chaque pas de temps, ce qui permet de capturer les dépendances séquentielles dans les données.
    """)
    st.image("recurrent_neural_network.png", use_column_width=True)

st.write('# Projet Deep Learning')
st.write('## Auteurs : Guy BATOLA, Yann OYE, Idrissa Belem, Alimatou DIOP')
# Zone de texte pour saisir le commentaire
comment = st.text_area("Saisissez votre commentaire ici")

# Bouton pour soumettre le commentaire
if st.button("Valider"):
    if comment:
        # Placeholder pour la prédiction du sentiment
        # Ici, vous pouvez ajouter la logique pour prédire le sentiment en fonction du modèle sélectionné
        # Par exemple, vous pouvez appeler une fonction qui utilise le modèle sélectionné pour prédire le sentiment du commentaire
        # Puis, afficher le résultat de la prédiction (positif, négatif ou neutre)
        st.write("Sentiment prédit : Positif")
    else:
        st.warning("Veuillez saisir un commentaire.")

