
import streamlit as st
import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np




st.title('To Grant or Not To Grant')



st.header('Input your data here',divider="red")
with st.expander('Prediction'):

    Accident_Date = st.date_input("Accident Date", datetime.date(2024, 12, 11))

    Claim_Identifier = st.number_input('Claim Identifier')

    Age_at_Injury = st.slider("Age at Injury", 0,117, 42)

    Alternative_Dispute_Resolution = st.selectbox(" Alternative Dispute Resolution",("Y", "N ", "U"))

    Average_Weekly_Wage = st.number_input("Average Weekly Wage")

    Birth_Year = st.slider("Birth Year", 1886, 2018, 1977)

    IME_4_Count = st.slider("IME-4 Count", 1, 73, 2)

    Industry_Code = st.slider("Industry Code",11, 92, 61)

    WCIO_Cause_of_Injury_Code = st.slider("WCIO Cause of Injury Code", 1, 99, 56)

    WCIO_Nature_of_Injury_Code = st.slider(" WCIO Nature of Injury Code",1, 91, 49)

    WCIO_Part_Of_Body_Code = st.slider("WCIO Part Of Body Code",1, 99, 38)
#10
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
    
    WCIO_Nature_of_Injury_Description = st.selectbox("WCIO Nature of Injury Description"
    , ('CONTUSION', 'SPRAIN OR TEAR','CONCUSSION',
     'PUNCTURE', 'LACERATION', 'ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC',
     'ALL OTHER SPECIFIC INJURIES, NOC', 'INFLAMMATION', 'BURN', 'STRAIN OR TEAR',
     'FRACTURE', 'FOREIGN BODY', 'MULTIPLE PHYSICAL INJURIES ONLY', 'RUPTURE', 'DISLOCATION',
     'ALL OTHER CUMULATIVE INJURY, NOC','HERNIA', 'ANGINA PECTORIS', 'CARPAL TUNNEL SYNDROME',
     'NO PHYSICAL INJURY', 'INFECTION', 'CRUSHING', 'SYNCOPE', 'POISONING - GENERAL (NOT OD OR CUMULATIVE',
     'RESPIRATORY DISORDERS', 'HEARING LOSS OR IMPAIRMENT', 'MENTAL STRESS',
     'SEVERANCE', 'ELECTRIC SHOCK', 'LOSS OF HEARING', 'DUST DISEASE, NOC', 'DERMATITIS',
     'ASPHYXIATION', 'MENTAL DISORDER', 'CONTAGIOUS DISEASE', 'AMPUTATION', 'MYOCARDIAL INFARCTION',
     'POISONING - CHEMICAL, (OTHER THAN METALS)','MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL',
     'VISION LOSS', 'VASCULAR', 'COVID-19', 'CANCER', 'HEAT PROSTRATION','AIDS', 'ENUCLEATION',
     'ASBESTOSIS', 'POISONING - METAL', 'VDT - RELATED DISEASES', 'FREEZING', 'BLACK LUNG', 'SILICOSIS',
     'ADVERSE REACTION TO A VACCINATION OR INOCULATION', 'HEPATITIS C', 'RADIATION', 'BYSSINOSIS'))

    WCIO_Part_Of_Body_Description = st.selectbox("WCIO Part Of Body Description", 
    ('BUTTOCKS', 'SHOULDER(S)', 'MULTIPLE HEAD INJURY', 'FINGER(S)', 'LUNGS', 'EYE(S)',
     'ANKLE', 'KNEE', 'THUMB', 'LOWER BACK AREA', 'ABDOMEN INCLUDING GROIN', 'LOWER LEG',
     'HIP', 'UPPER LEG', 'MOUTH', 'WRIST', 'SPINAL CORD', 'HAND', 'SOFT TISSUE', 'UPPER ARM',
     'FOOT', 'ELBOW', 'MULTIPLE UPPER EXTREMITIES', 'MULTIPLE BODY PARTS (INCLUDING BODY',
     'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS', 'MULTIPLE NECK INJURY', 'CHEST', 'WRIST (S) & HAND(S)', 'EAR(S)',
     'MULTIPLE LOWER EXTREMITIES', 'DISC', 'LOWER ARM', 'MULTIPLE', 'UPPER BACK AREA','SKULL',
     'TOES', 'FACIAL BONES', 'TEETH', 'NO PHYSICAL INJURY', 'MULTIPLE TRUNK', 'WHOLE BODY',
     'INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED','PELVIS', 'NOSE', 'GREAT TOE',
     'INTERNAL ORGANS', 'HEART', 'VERTEBRAE','LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA',
     'BRAIN', 'SACRUM AND COCCYX', 'ARTIFICIAL APPLIANCE', 'LARYNX', 'TRACHEA'))

    ZipCode = st.number_input("Enter your zip code:", value=None, placeholder="Type a number...")

    WCB_Decision = st.selectbox("WCB Decision",('Not Work Related'))

    OIICS_Nature_of_Injury_Description = st.text_input('OIICS Nature of Injury Description')

    data ={'Accident Date': Accident_Date, 'Age at Injury': Age_at_Injury, 'Alternative Dispute Resolution': Alternative_Dispute_Resolution,
       'Assembly Date': Assembly_Date,'Attorney/Representative': Attorney_Representative, 'Average Weekly Wage': Average_Weekly_Wage,
       'Birth Year': Birth_Year, 'C-2 Date': C2_Date, 'C-3 Date': C3_Date, 'Carrier Name': Carrier_Name, 'Carrier Type': Carrier_Type,
       'Claim Identifier': Claim_Identifier, 'County of Injury': County_of_Injury, 'COVID-19 Indicator': COVID_19_Indicator,
       'District Name': District_Name, 'First Hearing Date': First_Hearing_Date, 'Gender': Gender, 'IME-4 Count': IME_4_Count,
       'Industry Code': Industry_Code, 'Industry Code Description': Industry_Code_Description, 'Medical Fee Region': Medical_Fee_Region,
       'OIICS Nature of Injury Description': OIICS_Nature_of_Injury_Description, 'WCIO Cause of Injury Code': WCIO_Cause_of_Injury_Code,
       'WCIO Cause of Injury Description': WCIO_Cause_of_Injury_Description, 'WCIO Nature of Injury Code': WCIO_Nature_of_Injury_Code,
       'WCIO Nature of Injury Description': WCIO_Nature_of_Injury_Description ,'WCIO Part Of Body Code': WCIO_Part_Of_Body_Code,
       'WCIO Part Of Body Description': WCIO_Part_Of_Body_Description,'Zip Code': ZipCode, 'Number of Dependents': Number_of_Dependents}
   
   
    test = pd.DataFrame(data, index =[0]) 

    train = pd.read_csv('train_data_mok.csv')

    train.set_index('Claim Identifier', inplace=True)
    test.set_index('Claim Identifier', inplace=True)

    train.dropna(subset=['Claim Injury Type'], inplace=True)

    train.drop('OIICS Nature of Injury Description', axis = 1, inplace = True)
    test.drop('OIICS Nature of Injury Description', axis = 1, inplace = True)

    train.drop('WCB Decision', axis = 1, inplace = True)

    train['Birth Year'] = train['Birth Year'].replace(0, np.nan)


    train['IME-4 Count'] = train['IME-4 Count'].fillna(0)
    test['IME-4 Count'] = test['IME-4 Count'].fillna(0)

    train["Industry Code"] = train["Industry Code"].fillna(0)
    test["Industry Code"] = test["Industry Code"].fillna(0)

    train["Industry Code Description"] = train["Industry Code Description"].fillna('Not Aplicable')
    test["Industry Code Description"] = test["Industry Code Description"].fillna('Not Aplicable')

    train["WCIO Cause of Injury Code"] = train["WCIO Cause of Injury Code"].fillna(0)
    test["WCIO Cause of Injury Code"] = test["WCIO Cause of Injury Code"].fillna(0)

    train["WCIO Cause of Injury Description"] = train["WCIO Cause of Injury Description"].fillna('Not Aplicable')
    test["WCIO Cause of Injury Description"] = test["WCIO Cause of Injury Description"].fillna('Not Aplicable')

    train["WCIO Nature of Injury Code"] = train["WCIO Nature of Injury Code"].fillna(0)
    test["WCIO Nature of Injury Code"] = test["WCIO Nature of Injury Code"].fillna(0)

    train["WCIO Nature of Injury Description"] = train["WCIO Nature of Injury Description"].fillna('Not Aplicable')
    test["WCIO Nature of Injury Description"] = test["WCIO Nature of Injury Description"].fillna('Not Aplicable')

    train["WCIO Part Of Body Code"] = train["WCIO Part Of Body Code"].fillna(0)
    test["WCIO Part Of Body Code"] = test["WCIO Part Of Body Code"].fillna(0)

    train["WCIO Part Of Body Description"] = train["WCIO Part Of Body Description"].fillna('Not Aplicable')
    test["WCIO Part Of Body Description"] = test["WCIO Part Of Body Description"].fillna('Not Aplicable')

    train['Accident Date'] = pd.to_datetime(train['Accident Date']).dt.date
    train['Assembly Date'] = pd.to_datetime(train['Assembly Date']).dt.date
    train["C-2 Date"] = pd.to_datetime(train["C-2 Date"]).dt.date
    train["C-3 Date"] = pd.to_datetime(train["C-3 Date"]).dt.date
    train["First Hearing Date"] = pd.to_datetime(train["First Hearing Date"]).dt.date

    test['Accident Date'] = pd.to_datetime(test['Accident Date']).dt.date
    test['Assembly Date'] = pd.to_datetime(test['Assembly Date']).dt.date
    test["C-2 Date"] = pd.to_datetime(test["C-2 Date"]).dt.date
    test["C-3 Date"] = pd.to_datetime(test["C-3 Date"]).dt.date
    test["First Hearing Date"] = pd.to_datetime(test["First Hearing Date"]).dt.date

    train['Average Weekly Wage'] = np.log10(train['Average Weekly Wage'] + 1)
    test['Average Weekly Wage'] = np.log10(test['Average Weekly Wage'] + 1)

    def create_lookup(dataframe, code_column, description_column):
    lookup = dataframe[[code_column, description_column]].drop_duplicates()
    lookup = lookup.reset_index(drop=True)
    lookup[code_column] = lookup[code_column].astype('Int64')
    lookup = lookup.sort_values(by=code_column).reset_index(drop=True)
    
    return lookup


    WCIO_cause_Lookup = create_lookup(train, 'WCIO Cause of Injury Code', 'WCIO Cause of Injury Description')
    WCIO_nature_Lookup = create_lookup(train, 'WCIO Nature of Injury Code', 'WCIO Nature of Injury Description')
    WCIO_bodyPart_Lookup = create_lookup(train, 'WCIO Part Of Body Code', 'WCIO Part Of Body Description')
    IndustryCode_Lookup = create_lookup(train, 'Industry Code', 'Industry Code Description')


    train = train.drop(columns=['WCIO Nature of Injury Description'])
    test = test.drop(columns=['WCIO Nature of Injury Description'])

    train = train.drop(columns=['WCIO Cause of Injury Description'])
    test = test.drop(columns=['WCIO Cause of Injury Description'])

    train = train.drop(columns=['WCIO Part Of Body Description'])
    test = test.drop(columns=['WCIO Part Of Body Description'])

    train = train.drop(columns=['Industry Code Description'])
    test = test.drop(columns=['Industry Code Description'])

    x = train.drop(columns= 'Claim Injury Type')
    y = train['Claim Injury Type']

    

    

     model = XGBClassifier(
                objective='multi:softprob',
                random_state=42,
                learning_rate=0.05046195857265063,
                max_depth=14,
                min_child_weight=4.295663382738008,
                subsample=0.5794673021390964,
                colsample_bytree=0.6752893520492427,
                n_estimators=716,
                reg_alpha=0.2207882375290882,
                reg_lambda=0.28791727579162424,
                gamma=1.5556906330098323,
    )

  if st.button('Predict'):
                model.fit(x,y)
                prediction = model.predict([[test]])
                st.write(f'Prediction: {prediction[0]}')
