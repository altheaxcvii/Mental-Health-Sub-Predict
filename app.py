#importing and downloading 
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_chat import message
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Configurations
st.set_page_config(page_title='Subreddit Prediction', page_icon='🔮')
st.title('🔮Subreddit Prediction🔮')
st.write('This machine learning model is built with xxx model(s) and will predict whether the post comes from r/bipolar or r/schizophrenia')


#loading model
with open("testvotemodel.pkl", "rb") as file:
    loaded_model = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
    loaded_vectorizer = pickle.load(file)

#Creating functions
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
    X = loaded_vectorizer.transform(aseries)
    return loaded_model.predict(X)

def download_data():
    data = 'Disclaimer: The machine outputs are predictions and should NOT be used for diagnosis. \n\n ----------------------\n\n'
    for i in range(len(st.session_state['machine'])):
        data += f"User: {st.session_state['user'][i]}\n\n\n"
        data += f"Machine: {st.session_state['machine'][i]}\n\n\n\n\n"
    return data


def convo():
    if 'input_key' not in st.session_state:
        st.session_state['input_key'] = 0

    user_input = st.text_input("Insert a reddit post from either r/bipolar or r/schizophrenia", key='input_' + str(st.session_state['input_key']))

    if st.button('Submit', key='button_' + str(st.session_state['input_key'])):
        if len(user_input) > 0:
            st.session_state.machine.append(f'I think the post is from r/{predict(user_input)[0]}')
            st.session_state.user.append(user_input)
            st.session_state['input_key'] += 1  # create a new input box for the next run
        else:
            st.session_state.machine.append('I cannot predict if you don\'t send me anything')
            st.session_state.user.append('Too lazy to write anything')
            st.session_state['input_key'] += 1  # create a new input box for the next run

        st.experimental_rerun()  # Rerun the app to display the updated messages


#Sidebar
with st.sidebar:
    st.subheader('How to Use')
    st.write('1. Insert a reddit post\'s title and text.')
    st.write('2. Click the Submit button.')
    st.write('3. You may download the entire conversation as a .txt file by using the button below')
    data = download_data()
    st.download_button(
        label="Download Conversation Data",
        data=data,
        file_name="conversation_data.txt",
        mime="text/plain"
    )
    st.markdown('''---''')
    st.subheader('About Us')
    st.write('We are the Data Regressionantics from GA DSI37😎')
    st.markdown('''---''')
    st.subheader('⚠️Disclaimer⚠️')
    st.write('Not for diagnostic usage!! Please find a legit psychiatrist and get them to use their holy DSM V to diagnose you. Thanks!!')
    st.markdown('''---''')
    st.subheader('FAQ')
    st.subheader('Is it 100% accurate?')
    st.write('No, the machine learning model is not 100% accurate. While it is designed to make predictions and perform tasks to the best of its abilities, it is important to note that it may still make errors or provide imperfect results. Machine learning models are trained based on existing data and patterns, and their accuracy depends on the quality and diversity of the data used for training. Additionally, the model\'s performance can be influenced by various factors such as the complexity of the task, the availability of relevant data, and the limitations of the model itself. Therefore, it is advisable to interpret the model\'s outputs with a level of caution and consider its predictions as probabilistic rather than absolute certainties. Regular monitoring, evaluation, and improvement of the model are crucial to enhance its accuracy over time.')
    st.subheader('What is the purpose of your model?')
    st.write('The model aims to predict whether the posts comes from r/bipolar or r/schizophrenia, based on their content.')
    st.subheader('What data did you use to train your model?')
    st.write('We used existing posts from both subreddit that we scrapped to train our model.')
    st.subheader('What should I do if I need mental health help?')
    st.write('If you are facing a mental health crisis, please call SG Mental Health Helpline by IMH at +65 6389 2222')
    st.subheader('Is user privacy ensured?')
    st.write('Any text that are inputted into the model here are NOT saved (because I\'m not good enough to code that in yet🥲)')


#Start of 'chat bot' codes PLEASE DO NOT EDIT ANYTHING BELOW
message('Hello! I will predict if you had sent me a post from r/bipolar or r/schizophrenia. I may not be 100% accurate, but I do try my best!')

if 'machine' not in st.session_state:
    st.session_state['machine'] = []
    st.session_state['user'] = []

if st.session_state['user']:
    for i in range(len(st.session_state['machine'])):
        message(st.session_state['user'][i], is_user=True, key=f"user_message_{i}")
        message(st.session_state['machine'][i], key=f"machine_message_{i}")
    convo()
else:
    convo()
