import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np


nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
# Liste des modèles disponibles
MODELS = ['Réseaux de neurones recurrent', 'SentimentIntensityAnalyzer']

# Chargement du vocabulaire et des modèles pré-entrainés
vectorize = pickle.load(open("vectorizer.pkl", "rb"))
top_features_boolean = pickle.load(open("topFeature.pkl", "rb"))
rnn = pickle.load(open("RNN.pkl", "rb"))

def sentiment_predit(text):
    sentiment_scores = sia.polarity_scores(text)
    lst = list(sentiment_scores.values())[0:3]
    max_value = np.max(lst)
    index_val = lst.index(max_value)
    if index_val==0:
      return ("Negatif", lst)
    elif index_val==1:
      return ("Mitige", lst)
    else:
      return ("Positif", lst)

# Sélection du modèle
selected_model = st.sidebar.selectbox("Sélectionnez un modèle", MODELS)

# Affichage de l'explication du modèle sélectionné
if selected_model == 'Réseaux de neurones recurrent':
    st.sidebar.markdown("""
    Le réseau de neurones avec Word Embedding utilise une couche d'embedding pour convertir les mots en vecteurs denses avant de les passer au réseau de neurones. Cela permet au modèle de capturer les relations sémantiques entre les mots.
    """)
   
elif selected_model == 'SentimentIntensityAnalyzer':
    st.sidebar.markdown("""
    SentimentIntensityAnalyzer est un outil de l'outil de traitement du langage naturel (NLP) de NLTK pour analyser les sentiments dans un texte en attribuant des scores de polarité. Il fonctionne en attribuant des valeurs de positivité, 
    négativité, neutralité et composée à chaque phrase en se basant sur des mots-clés et des règles prédéfinis. Ces scores sont calculés en pondérant les mots du texte selon leur contribution à la polarité du texte, permettant ainsi de 
    déterminer le sentiment global du texte.
    """)
    

st.write('# Projet Deep Learning')
st.write('## Auteurs : Guy BATOLA, Yann OYE, Idrissa Belem, Alimatou DIOP')
st.write('### Objectif : ')
st.write("""Construction d'un modèle capable de prédire le sentiment sur "Fine Foods" et le déployer sous forme de service web pour un client.
            Nous avons utilisé notre propore modèle de deep learning utilisant des réseaux de neuronnes récurrents. En bonus, nous avons intégré
            un autre modèle de NLP, préentrainé du package ntlk
        """)

# Zone de texte pour saisir le commentaire
comment = st.text_area("Saisissez votre commentaire ici")

# Bouton pour soumettre le commentaire
if st.button("Valider"):
    if comment:
        # RNN
        if selected_model == 'Réseaux de neurones recurrent' :
            a = vectorize.transform([comment])
            proba = rnn.predict(a.toarray()[:,top_features_boolean])
            sentiment = proba.argmax(-1)
            if sentiment == 0:
                st.write("Sentiment prédit : ",  "Negatif")
            elif sentiment == 2:
                st.write("Sentiment prédit : ",  "Positif")
            else :
                st.write("Sentiment prédit : ",  "Mitige")
                
            st.write("Probabilité que le message écrit soit positif : ", probas[0, 2])
            st.write("Probabilité que le message écrit soit négatif : ", probas[0, 0])
            st.write("Probabilité que le message écrit soit neutre : ", probas[0, 1])
        else:
            reponse = sentiment_predit(comment)
            sentiment = reponse[0]
            probas = reponse[1]
            st.write("Sentiment prédit : ", sentiment)

            st.write("Probabilité que le message écrit soit positif : ", probas[2])
            st.write("Probabilité que le message écrit soit négatif : ", probas[0])
            st.write("Probabilité que le message écrit soit neutre : ", probas[1])
    else:
        st.warning("Veuillez saisir un commentaire.")
