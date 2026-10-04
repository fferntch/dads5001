import streamlit as st
import pandas as pd
import numpy as np
import requests
from transformers import pipeline

st.title('Uber pickups in NYC by fern')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL,nrows=nrows)
    st.write(data.shape)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = load_data(10000)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Done! (using st.cache_data)')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour,
    bins=24,
    range=(0,24)
)[0]
#st.write(hist_values)
st.bar_chart(hist_values)

#st.subheader('Map of all pickups')
#st.map(data)
# Keep the slider and the number box in sync via session_state.
st.session_state.setdefault('hour_slider', 17)
st.session_state.setdefault('hour_input', 17)

def sync_from_slider():
    st.session_state.hour_input = st.session_state.hour_slider

def sync_from_input():
    st.session_state.hour_slider = st.session_state.hour_input

col1, col2 = st.columns([3, 1])
with col1:
    st.slider('hour', 0, 23, key='hour_slider', on_change=sync_from_slider)
with col2:
    st.number_input('type hour', 0, 23, step=1, key='hour_input', on_change=sync_from_input)

hour_to_filter = st.session_state.hour_slider
filtered_data = data[data[DATE_COLUMN].dt.hour==hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

@st.cache_data
def api_call():
    response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
    return response.json()

ans = api_call()
st.write(ans)

st.subheader("Transformers")

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    #return pipeline("sentiment-analysis")
    #return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    return pipeline("text-classification", model="FlukeTJ/distilbert-base-thai-sentiment")

model = load_model()

query = st.text_input("Your query", value="I love Streamlit! 🎈")
if query:
    result = model(query)[0]  # 👈 Classify the query text
    st.write(result)
    #st.write(result['label'])

