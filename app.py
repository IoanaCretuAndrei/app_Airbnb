import streamlit as st 
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model('ml_airbnb')
st.title('Sistema de predicción de precios Airbnb-TORONTO')

neighbourhood = st.selectbox('Barrio', options = ['Little Portugal', 'Waterfront Communities-The Island',
    'Woodbine Corridor', 'South Riverdale', 'South Parkdale',
    'The Beaches', 'Junction Area', 'Wexford/Maryvale',
    'Rosedale-Moore Park', 'Morningside', 'Kensington-Chinatown',
    'Bay Street Corridor', 'East End-Danforth',
    'Church-Yonge Corridor', 'Niagara',
    'Cabbagetown-South St.James Town', 'High Park North',
    'High Park-Swansea', 'Woburn', 'Roncesvalles', 'Moss Park',
    'Oakridge', 'Don Valley Village', 'Annex', 'Caledonia-Fairbank',
    'Casa Loma', 'North St.James Town', 'Thistletown-Beaumond Heights',
    'Dovercourt-Wallace Emerson-Junction', 'Blake-Jones',
    'Palmerston-Little Italy', 'Willowdale East', 'Danforth',
    'University', 'Flemingdon Park', 'Forest Hill South',
    'Brookhaven-Amesbury', 'Oakwood Village', 'Newtonbrook West',
    'Trinity-Bellwoods', 'Playter Estates-Danforth',
    'Greenwood-Coxwell', 'Mimico (includes Humber Bay Shores)',
    'Regent Park', 'Dufferin Grove', 'Birchcliffe-Cliffside',
    'Broadview North', 'North Riverdale', 'Humewood-Cedarvale',
    'Mount Pleasant East', 'Parkwoods-Donalda', 'Yonge-St.Clair',
    'Old East York', 'Mount Dennis', 'Agincourt South-Malvern West',
    'Willowdale West', 'Islington-City Centre West',
    'Corso Italia-Davenport', 'St.Andrew-Windfields',
    'Stonegate-Queensway', 'Yonge-Eglinton', 'Rockcliffe-Smythe',
    'Lawrence Park North', 'Bendale', 'Clanton Park',
    'Englemount-Lawrence', 'Mount Pleasant West',
    'Bayview Woods-Steeles', 'New Toronto', 'Bayview Village',
    'Weston-Pellam Park', 'Cliffcrest', 'Guildwood', 'Agincourt North',
    'Alderwood', "L'Amoreaux", 'Lambton Baby Point',
    'Woodbine-Lumsden', 'Danforth East York',
    'Bridle Path-Sunnybrook-York Mills', 'Wychwood',
    'Etobicoke West Mall', 'Runnymede-Bloor West Village',
    'Bedford Park-Nortown', "Tam O'Shanter-Sullivan",
    'Lansing-Westgate', 'Long Branch', 'York University Heights',
    'Briar Hill-Belgravia', 'Westminster-Branson',
    'Lawrence Park South', 'Leaside-Bennington',
    'Keelesdale-Eglinton West', 'Hillcrest Village', 'Bathurst Manor',
    'Kingsway South', 'Ionview', 'Weston', 'Yorkdale-Glen Park',
    'Pelmo Park-Humberlea', 'Clairlea-Birchmount',
    'Eringate-Centennial-West Deane', 'Eglinton East',
    'West Humber-Clairville', 'West Hill', 'Kennedy Park',
    'Edenbridge-Humber Valley', 'Rouge', 'Black Creek',
    'Willowridge-Martingrove-Richview', 'Beechborough-Greenbrook',
    'Rexdale-Kipling', 'Pleasant View', "O'Connor-Parkview",
    'Victoria Village', 'Scarborough Village', 'Henry Farm',
    'Markland Wood', 'Princess-Rosethorn', 'Banbury-Don Mills',
    'Dorset Park', 'Milliken', 'Kingsview Village-The Westway',
    'Thorncliffe Park', 'Malvern', 'Newtonbrook East',
    'Mount Olive-Silverstone-Jamestown', 'Glenfield-Jane Heights',
    'Highland Creek', 'Taylor-Massey', 'Elms-Old Rexdale',
    'Forest Hill North', 'Steeles', 'Downsview-Roding-CFB',
    'Maple Leaf', 'Humbermede', 'Humber Heights-Westmount',
    'Centennial Scarborough', 'Humber Summit', 'Rustic'])

property_type = st.selectbox('Tipo de propiedad', options = ['Entire home', 'Private room in rental unit', 'Entire condo',
    'Private room in home', 'Entire rental unit', 'Entire guest suite',
    'Private room in condo', 'Entire loft', 'Entire townhouse',
    'Entire serviced apartment', 'Private room in townhouse',
    'Shared room in rental unit', 'Private room in tiny home',
    'Private room in guest suite', 'Entire guesthouse',
    'Private room in cottage', 'Private room in loft',
    'Entire bungalow', 'Private room in bungalow', 'Private room',
    'Shared room in home', 'Private room in guesthouse',
    'Shared room in condo', 'Entire villa', 'Entire place',
    'Private room in bed and breakfast', 'Shared room in townhouse',
    'Private room in barn', 'Shared room in bed and breakfast',
    'Private room in floor', 'Tiny home', 'Floor',
    'Private room in villa', 'Shared room in bungalow',
    'Shared room in hostel', 'Entire cottage',
    'Private room in castle', 'Shared room in loft', 'Entire home/apt',
    'Private room in hostel', 'Shared room in guesthouse', 'Camper/RV',
    'Room in boutique hotel', 'Earthen home',
    'Room in bed and breakfast', 'Shared room in boat',
    'Room in hotel', 'Private room in serviced apartment',
    'Private room in earthen home', 'Boat',
    'Shared room in guest suite', 'Private room in hut', 'Island',
    'Private room in casa particular', 'Room in aparthotel',
    'Entire vacation home', 'Private room in vacation home',
    'Shared room in serviced apartment', 'Castle',
    'Shipping container', 'Treehouse', 'Shared room in hotel',
    'Shared room in tiny home'])

room_type = st.selectbox('Tipo de habitación', options = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])

accommodates = st.slider('Numero de Personas', min_value=1, max_value=17, value=1 )
maximum_nights = st.slider('Estancia máxima', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Estancia mínima', min_value=1, max_value=10, value=1)


imput_data = pd.DataFrame ([[
    neighbourhood, property_type, accommodates, room_type, maximum_nights, minimum_nights
]], columns = ['neighbourhood','property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights' ])

if st.button('Descubre el precio!'):
    prediction = predict_model(model, data = imput_data)
    st.write(str(prediction['prediction_label'].values[0].round(2))+'euros')