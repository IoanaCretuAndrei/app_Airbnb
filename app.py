import streamlit as st 
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model('ml_airbnb')
st.title('Sistema de predicción de precios Airbnb')

neighbourhood = st.selectbox('Barrio', options = ['Oostelijk Havengebied - Indische Buurt', 'Westerpark','Centrum-Oost', 'Centrum-West', 'Bos en Lommer', 'Zuid','De Pijp - Rivierenbuurt', 'De Baarsjes - Oud-West', 'Oud-Oost','Slotervaart', 'Gaasperdam - Driemond', 'Oud-Noord', 'Noord-Oost','Watergraaf meer', 'IJburg - Zeeburgereiland','Geuzenveld - Slotermeer', 'Buitenveldert - Zuidas',
    'Bijlmer-Oost', 'Noord-West', 'De Aker - Nieuw Sloten', 'Osdorp',
    'Bijlmer-Centrum'])

property_type = st.selectbox('Tipo de propiedad', options = ['Apartment', 'Townhouse', 'Houseboat', 'Bed and breakfast', 'Boat',
    'Guest suite', 'Loft', 'Serviced apartment', 'House',
    'Boutique hotel', 'Guesthouse', 'Other', 'Condominium', 'Chalet',
    'Nature lodge', 'Tiny house', 'Hotel', 'Villa', 'Cabin',
    'Lighthouse', 'Bungalow', 'Hostel', 'Cottage', 'Tent',
    'Earth house', 'Campsite', 'Castle', 'Camper/RV', 'Barn',
    'Casa particular (Cuba)', 'Aparthotel'])

room_type = st.selectbox('Tipo de habitación', options = ['Private room', 'Entire home/apt', 'Shared room'])

accommodates = st.slider('Numero de Personas', min_value=1, max_value=17, value=1 )
maximum_nights = st.slider('Estancia máxima', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Estancia mínima', min_value=1, max_value=10, value=1)


imput_data = pd.DataFrame ([[
    neighbourhood, property_type, accommodates, room_type, maximum_nights, minimum_nights
]], columns = ['neighbourhood','property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights' ])

if st.button('Descubre el precio!'):
    prediction = predict_model(model, data = imput_data)
    st.write(str(prediction['prediction_label'].values[0].round(2))+'euros')