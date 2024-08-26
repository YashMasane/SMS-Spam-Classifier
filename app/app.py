import streamlit as st
from ui import make_prediction

st.title("Spam SMS Classifier")

sms = st.text_area("Write message you got here")

if st.button("predict"):
    if len(sms) != 0:
        pred = make_prediction(sms)
        if pred == 1:
            st.error('Spam')
        else:
            st.success('Not Spam')    
    else:
        st.write('Please enter a SMS first')        

