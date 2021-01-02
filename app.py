import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
from keras.layers import *
from keras.models import * 
from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
from keras.preprocessing import image
PAGE_CONFIG={"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


@st.cache(ttl=60*5,max_entries=20)



  






def load_data():
    data=pd.read_csv("https://api.covid19india.org/csv/latest/state_wise.csv")
    return data


data=load_data()


st.markdown('<style>description{colr:blue;}</style>',unsafe_allow_html=True)
st.title('🦠Covid-19 Impact in India Hello')
st.markdown("<description>The objective of this website is to offer an ongoing assessment of"+"COVID-19's impact in India.This website gives you the real time impact analysis of confirmed,"+"active,recovered,and death cases of Covid-19 on National-,State-,and District-level basis."+"The website's data is updated every 5 minutes in order to ensure the delivery of true and"+"accurate data.</description>",unsafe_allow_html=True)
st.sidebar.title('Select the parameters to analyze Covid-19 situation')

st.sidebar.checkbox("Show Analysis by State", True, key=1)
select = st.sidebar.selectbox('Select a State',data['State'])
#get the state selected in the selectbox
state_data = data[data['State'] == select]
select_status = st.sidebar.radio("Covid-19 patient's status", ('Confirmed',
'Active', 'Recovered', 'Deceased'))



def get_total_dataframe(dataset):
    total_dataframe = pd.DataFrame({
    'Status':['Confirmed', 'Active', 'Recovered', 'Deaths'],
    'Number of cases':(dataset.iloc[0]['Confirmed'],
    dataset.iloc[0]['Active'], dataset.iloc[0]['Recovered'],
    dataset.iloc[0]['Deaths'])})
    return total_dataframe
state_total = get_total_dataframe(state_data)
if st.sidebar.checkbox("Show Analysis by State", True, key=2):
    st.markdown("## **State level analysis**")
    st.markdown("### Overall Confirmed, Active, Recovered and " +
    "Deceased cases in %s yet" % (select))
    if not st.checkbox('Hide Graph', False, key=1):
        state_total_graph = px.bar(
        state_total,
        x='Status',
        y='Number of cases',
        labels={'Number of cases':'Number of cases in %s' % (select)},
        color='Status')
        st.plotly_chart(state_total_graph)



def get_table():
    datatable = data[['State', 'Confirmed', 'Active', 'Recovered', 'Deaths']].sort_values(by=['Confirmed'], ascending=False)
    datatable = datatable[datatable['State'] != 'State Unassigned']
    return datatable
datatable = get_table()
st.markdown("### Covid-19 cases in India")
st.markdown("The following table gives you a real-time analysis of the confirmed, active, recovered and deceased cases of Covid-19 pertaining to each state in India.")
#st.dataframe(datatable) # will display the dataframe
st.table(datatable)# will display the table

if st.checkbox(label="Predict?"):
  model = keras.models.load_model('model_final.h5')
  buffer = st.file_uploader("Image here pl0x")
  temp_file = NamedTemporaryFile(delete=False)
  if buffer:
    temp_file.write(buffer.getvalue())
    st.image(load_img(temp_file.name),caption='Uploaded x-ray image ',use_column_width=True)
    img=image.load_img(temp_file.name,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    if(pred[0][0]==1):
      st.write("Normal")

    else:
      st.write("Covid")
    
