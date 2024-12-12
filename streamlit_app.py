import streamlit as st
import pickle as pkl
import pandas
import datetime

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


st.title('To Grant or Not To Grant')

try:
    model = load_pickle('model.pkl')
except Exception as e:
    st.error(f"Error loading pickle file: {e}")

st.header('Input your data here',divider="red")
with st.expander('Prediction'):
    
    Accident_Date = st.date_input("Accident Date", datetime.date(2024, 12, 11))

    Age_at_Injury = st.slider("Age at Injury", 0,117, 42)

    Alternative_Dispute_Resolution = st.selectbox(" Alternative Dispute Resolution",("Y", "N ", "U"))

    Average_Weekly_Wage = st.number_input("Average Weekly Wage")

    Birth_Year = st.slider("Birth Year", 1886, 2018, 1977)

    IME_4_Count = st.slider("IME-4 Count", 1, 73, 2)

    Industry_Code = st.slider("Industry Code",11, 92, 61)

    WCIO_Cause_of_Injury_Code = st.slider("WCIO Cause of Injury Code", 1, 99, 56)

    WCIO_Nature_of_Injury_Code = st.slider(" WCIO Nature of Injury Code",1, 91, 49)

    WCIO_Part_Of_Body_Code = st.slider("WCIO Part Of Body Code",1, 99, 38)

    Agreement_Reached = st.slider("Agreement Reached",0,1)

    Number_of_Dependents = st.slider("Number of Dependents", 0, 6, 3)

    Assembly_Date = st.date_input("Assembly Date", datetime.date(2024, 12, 11))

    Attorney_Representative = st.selectbox("Attorney/Representative",("Y", "N"))

    C2_Date = st.date_input("  C-2 Date", datetime.date(2024, 12, 11))

    C3_Date = st.date_input("  C-3 Date", datetime.date(2024, 12, 11))

    Carrier_Name = st.text_input("Carrier Name", "STATE INSURANCE FUND")

    Carrier_Type = st.selectbox("Carrier Type",('1A. PRIVATE', '2A. SIF', '4A. SELF PRIVATE', '3A. SELF PUBLIC',
       'UNKNOWN', '5D. SPECIAL FUND - UNKNOWN',
       '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)',
       '5C. SPECIAL FUND - POI CARRIER WCB MENANDS'))

    County_of_Injury = st.text_input("County of Injury", "SUFFOLK")

    COVID_19_Indicator = st.selectbox("COVID_19_Indicator",("Y", "N"))

    District_Name = st.selectbox("DistrictName",('SYRACUSE', 'ROCHESTER', 'ALBANY', 'HAUPPAUGE', 'NYC', 'BUFFALO',
       'BINGHAMTON', 'STATEWIDE'))

    First_Hearing_Date = st.date_input("First Hearing Date", datetime.date(2024, 12, 11))

    Gender = st.selectbox("Gender",('M', 'F', 'U', 'X'))

    Industry_Code_Description = st.selectbox("Industry Code Description",('RETAIL TRADE', 'CONSTRUCTION',
       'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT',
       'HEALTH CARE AND SOCIAL ASSISTANCE',
       'ACCOMMODATION AND FOOD SERVICES', 'EDUCATIONAL SERVICES',
       'INFORMATION', 'MANUFACTURING', 'TRANSPORTATION AND WAREHOUSING',
       'WHOLESALE TRADE', 'REAL ESTATE AND RENTAL AND LEASING',
       'FINANCE AND INSURANCE',
       'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)',
       'PUBLIC ADMINISTRATION',
       'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES',
       'ARTS, ENTERTAINMENT, AND RECREATION', 'UTILITIES',
       'AGRICULTURE, FORESTRY, FISHING AND HUNTING', 'MINING',
       'MANAGEMENT OF COMPANIES AND ENTERPRISES'))
    Medical_Fee_Region = st.selectbox("Medical Fee Region ",('I', 'II','IV', 'UK', 'III'))

    WCIO_Cause_of_Injury_Description = st.selectbox("WCIO Cause of Injury Description ",(
'FROM LIQUID OR GREASE SPILLS', 'REPETITIVE MOTION', 'OBJECT BEING LIFTED OR HANDLED',
 'HAND TOOL, UTENSIL; NOT POWERED', 'FALL, SLIP OR TRIP, NOC', 'CUT, PUNCTURE, SCRAPE, NOC',
 'OTHER - MISCELLANEOUS, NOC', 'STRUCK OR INJURED, NOC', 'FALLING OR FLYING OBJECT', 'CHEMICALS',
 'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE','LIFTING', 'TWISTING', 'ON SAME LEVEL', 'STRAIN OR INJURY BY, NOC',
 'MOTOR VEHICLE, NOC', 'FROM DIFFERENT LEVEL (ELEVATION)','PUSHING OR PULLING', 'FOREIGN MATTER (BODY) IN EYE(S)',
 'FELLOW WORKER, PATIENT OR OTHER PERSON', 'STEAM OR HOT FLUIDS', 'STATIONARY OBJECT',
 'ON ICE OR SNOW', 'ABSORPTION, INGESTION OR INHALATION, NOC','PERSON IN ACT OF A CRIME', 'INTO OPENINGS', 'ON STAIRS',
 'FROM LADDER OR SCAFFOLDING', 'SLIP, OR TRIP, DID NOT FALL', 'JUMPING OR LEAPING', 'MOTOR VEHICLE',
 'RUBBED OR ABRADED, NOC','REACHING', 'OBJECT HANDLED', 'HOT OBJECTS OR SUBSTANCES',
 'ELECTRICAL CURRENT', 'HOLDING OR CARRYING', 'CAUGHT IN, UNDER OR BETWEEN, NOC', 'FIRE OR FLAME',
 'CUMULATIVE, NOC', 'POWERED HAND TOOL, APPLIANCE', 'STRIKING AGAINST OR STEPPING ON, NOC', 'MACHINE OR MACHINERY',
 'COLD OBJECTS OR SUBSTANCES', 'BROKEN GLASS', 'COLLISION WITH A FIXED OBJECT', 'STEPPING ON SHARP OBJECT',
 'OBJECT HANDLED BY OTHERS', 'DUST, GASES, FUMES OR VAPORS', 'OTHER THAN PHYSICAL CAUSE OF INJURY',
 'CONTACT WITH, NOC', 'USING TOOL OR MACHINERY', 'SANDING, SCRAPING, CLEANING OPERATION', 'CONTINUAL NOISE',
 'ANIMAL OR INSECT', 'MOVING PARTS OF MACHINE', 'GUNSHOT', 'WIELDING OR THROWING', 'MOVING PART OF MACHINE',
 'TEMPERATURE EXTREMES', 'HAND TOOL OR MACHINE IN USE', 'VEHICLE UPSET', 'COLLAPSING MATERIALS (SLIDES OF EARTH)',
 'TERRORISM', 'PANDEMIC', 'WELDING OPERATION', 'NATURAL DISASTERS', 'EXPLOSION OR FLARE BACK',
 'RADIATION', 'CRASH OF RAIL VEHICLE','MOLD', 'ABNORMAL AIR PRESSURE', 'CRASH OF WATER VEHICLE', 'CRASH OF AIRPLANE'))

    OIICS_Nature_of_Injury_Description = st.text_input("OIICS Nature of Injury Description")

    WCB_Decision = st.selectbox("WCB Decision",('Not Work Related'))

    

    
    
    
        
    

    
    
    
    
    

    

    
    
    

# Make a prediction using the model
    if st.button('Predict'):
        prediction = model.predict([[input_value]])
        st.write(f'Prediction: {prediction[0]}')
    
st.divider()
