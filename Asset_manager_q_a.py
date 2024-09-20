
'''
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the questions and answers from a JSON file
def load_qa_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Preprocess text by tokenizing and removing stopwords (in German)
def preprocess_text(text):
    stop_words = set(stopwords.words('german'))  # Use German stopwords
    words = word_tokenize(text.lower(), language='german')  # Tokenize using German language setting
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Find the most similar question in the dataset and return the corresponding answer
def get_response(user_input, qa_data, similarity_threshold=0.75):
    # Preprocess questions and user input
    questions = [preprocess_text(qa['Question']) for qa in qa_data]
    processed_input = preprocess_text(user_input)
    
    # Use TF-IDF Vectorizer to convert text to vectors
    vectorizer = TfidfVectorizer().fit_transform(questions + [processed_input])
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between user input and each question
    cosine_similarities = cosine_similarity([vectors[-1]], vectors[:-1])
    most_similar_index = cosine_similarities.argmax()
    highest_similarity = cosine_similarities[0, most_similar_index]
    
    # If similarity is below the threshold, suggest the closest question instead
    if highest_similarity < similarity_threshold:
        closest_question = qa_data[most_similar_index]['Question']
        return f"Ich verstehe nicht. Meinten Sie: '{closest_question}'?"
    else:
        # Get the most similar question's answer
        return qa_data[most_similar_index]['Answer']

# Main chatbot function
def chatbot():
    # Load QA data
    qa_data = load_qa_data('C:/Users/z004nkac/OneDrive - Siemens AG/Desktop/LLM/Test_Ground/Test/Stage_2_Playground/Asset_Manager_Q_A.json')  # Use the German JSON file
    
    print("Chatbot: Hallo! Ich bin hier, um zu helfen. Geben Sie 'exit' ein, um die Unterhaltung zu beenden.")
    
    while True:
        user_input = input("Du: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Auf Wiedersehen!")
            break
        
        # Get the chatbot response
        response = get_response(user_input, qa_data)
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
'''

import json
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the questions and answers from a JSON file
def load_qa_data(json_file):
    data = json.load(json_file)
    return data

# Preprocess text by tokenizing and removing stopwords (in German)
def preprocess_text(text):
    stop_words = set(stopwords.words('german'))  # Use German stopwords
    words = word_tokenize(text.lower(), language='german')  # Tokenize using German language setting
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Find the most similar question in the dataset and return the corresponding answer
def get_response(user_input, qa_data, similarity_threshold=0.75):
    # Preprocess questions and user input
    questions = [preprocess_text(qa['Question']) for qa in qa_data]
    processed_input = preprocess_text(user_input)
    
    # Use TF-IDF Vectorizer to convert text to vectors
    vectorizer = TfidfVectorizer().fit_transform(questions + [processed_input])
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between user input and each question
    cosine_similarities = cosine_similarity([vectors[-1]], vectors[:-1])
    most_similar_index = cosine_similarities.argmax()
    highest_similarity = cosine_similarities[0, most_similar_index]
    
    # If similarity is below the threshold, suggest the closest question instead
    if highest_similarity < similarity_threshold:
        closest_question = qa_data[most_similar_index]['Question']
        return f"Ich verstehe nicht. Meinten Sie: '{closest_question}'?"
    else:
        # Get the most similar question's answer
        return qa_data[most_similar_index]['Answer']

# Main Streamlit chatbot function
def chatbot():
    st.title("Chatbot - Asset Manager Q&A")

    # Upload QA JSON file
    json_file = st.file_uploader("Laden Sie die JSON-Datei hoch", type="json")
    
    if json_file is not None:
        qa_data = load_qa_data(json_file)
        st.success("Datei erfolgreich hochgeladen!")

        # Chat interface
        user_input = st.text_input("Stellen Sie eine Frage:")
        
        if st.button("Senden"):
            if user_input:
                # Get the chatbot response
                response = get_response(user_input, qa_data)
                st.write(f"Chatbot: {response}")
            else:
                st.write("Bitte geben Sie eine Frage ein.")
    else:
        st.write("Bitte laden Sie eine JSON-Datei hoch.")

# Run the Streamlit app
if __name__ == "__main__":
    chatbot()
