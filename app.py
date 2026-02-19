# import necessary library
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Spliting Data
from sklearn.model_selection import train_test_split

# Choosing Model & Training The Model
from sklearn.naive_bayes import MultinomialNB

# Load CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# Data Collection
data = pd.read_csv("dataset/spam.csv", encoding='latin-1')

# Feature Selection
data = data[["class", "message"]]
x = np.array(data["message"])
y = np.array(data["class"])

# Choosing Model & Training The Model
cv = CountVectorizer()
x = cv.fit_transform(x) # fit the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# another model 
clf = MultinomialNB()
clf.fit(x_train, y_train)

# import streamlit 
import streamlit as st
st.markdown("<h1>📧 Spam Detection System</h1>", unsafe_allow_html=True)

st.markdown("### Enter your SMS/E-mail below:")

def spamdetection():
    sample = st.text_area("", placeholder="Type your message here...")

    if st.button("Check Message"):
        if len(sample) < 1:
            st.warning("Please enter a message.")
        else:
            data = cv.transform([sample]).toarray()
            prediction = clf.predict(data)[0]

            if prediction == "spam":
                st.markdown(
                    '<div class="result-box" style="background-color:#ff4d4d;color:white;">🚫 This is SPAM</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-box" style="background-color:#2ecc71;color:white;">✅ This is NOT Spam</div>',
                    unsafe_allow_html=True
                )


spamdetection()