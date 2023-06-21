import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


st.title('🔮Subreddit Prediction🔮')
st.write('This machine learning model is built with xxx model(s) and will predict whether the post comes from r/bipolar or r/schizophrenia')


#loading model
with open("testvotemodel.pkl", "rb") as file:
    loaded_model = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
    loaded_vectorizer = pickle.load(file)

def preprocesstext(text):
    text = text.lower() #convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text) #remove symbols
    tokens = word_tokenize(text)
    stopword = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopword]
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def predict(astring):
    aseries = pd.Series(astring)
    aseries = aseries.apply(preprocesstext)
    with open("testvotemodel.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    with open("vectorizer.pkl", "rb") as file:
        loaded_vectorizer = pickle.load(file)
    X = loaded_vectorizer.transform(aseries)
    return loaded_model.predict(X)

loreumtext = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

st.subheader('Input a Reddit Post')
astring = st.text_area('Paste the post title and post content here.', placeholder = loreumtext, height = 200)
if st.button('Submit'):
    if astring.strip():
        st.write(f'The post is predicted to be from r/{predict(astring)[0]}')
        st.balloons()
        if predict(astring)[0] == 'schizophrenia':
            st.write('Schizophrenia is a chronic brain disorder that affects less than one percent of the U.S. population. When schizophrenia is active, symptoms can include delusions, hallucinations, disorganized speech, trouble with thinking and lack of motivation.')
        elif predict(astring)[0] == 'bipolar':
            st.write('Bipolar disorder (formerly called manic-depressive illness or manic depression) is a mental illness that causes unusual shifts in a person\'s mood, energy, activity levels, and concentration. These shifts can make it difficult to carry out day-to-day tasks.')
    else:
        st.warning('I told you to input some text, didn\'t I?', icon='⚠️')


with st.sidebar:
    st.subheader('How to Use')
    st.write('1. Insert a reddit post\'s title and text.')
    st.write('2. Click the Submit button.')
    st.markdown('''---''')
    st.subheader('About Us')
    st.write('random text to be edited')
    st.markdown('''---''')
    st.subheader('Disclaimer')
    st.write('Not for diagnostic usage!! Please find a legit psychiatrist and get them to use their holy DSM V to diagnose you. Thanks!!')
    st.markdown('''---''')
    st.subheader('FAQ')
    st.subheader('Is it 100% accurate?')
    st.write('No, the machine learning model is not 100% accurate. While it is designed to make predictions and perform tasks to the best of its abilities, it is important to note that it may still make errors or provide imperfect results. Machine learning models are trained based on existing data and patterns, and their accuracy depends on the quality and diversity of the data used for training. Additionally, the model\'s performance can be influenced by various factors such as the complexity of the task, the availability of relevant data, and the limitations of the model itself. Therefore, it is advisable to interpret the model\'s outputs with a level of caution and consider its predictions as probabilistic rather than absolute certainties. Regular monitoring, evaluation, and improvement of the model are crucial to enhance its accuracy over time.')
