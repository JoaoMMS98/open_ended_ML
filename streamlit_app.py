

import streamlit as st
import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import warnings
from sklearn.decomposition import PCA



st.title('To Grant or Not To Grant')

st.header('Input your data here',divider="red")
with st.expander('Prediction'):
    def engineer_features(df):
        # WCIO PCA
        wcio_features = ['WCIO Nature of Injury Code', 'WCIO Part Of Body Code', 'WCIO Cause of Injury Code']
        pca = PCA(n_components=2)
        wcio_pca = pca.fit_transform(StandardScaler().fit_transform(df[wcio_features]))
        df['wcio_pca1'] = wcio_pca[:, 0]
        df['wcio_pca2'] = wcio_pca[:, 1]

        # Region clustering
        df['region_cluster'] = df['Medical Fee Region'].astype(str) + '_' + df['Zip Code'].astype(str) + df['County of Injury'].astype(str)

        # Market indicator based on WCIO frequencies
        high_risk_nature = [20658, 12456, 28947]
        high_risk_body = [11229, 8780, 3028]
        high_risk_cause = [20096, 12441, 18015, 340, 1989]

        # Add high risk indicators
        df['high_risk_nature'] = df['WCIO Nature of Injury Code'].isin(high_risk_nature).astype(int)
        df['high_risk_body'] = df['WCIO Part Of Body Code'].isin(high_risk_body).astype(int)
        df['high_risk_cause'] = df['WCIO Cause of Injury Code'].isin(high_risk_cause).astype(int)

        df['market_indicator'] = ((df['high_risk_nature']) |
                                (df['high_risk_body']) |
                                (df['high_risk_cause'])).astype(int)
    
        # Drop redundant columns
        cols_to_drop = [] #'Birth Year', 'COVID-19 Indicator_Y' + wcio_features

        return df.drop(columns=cols_to_drop)
        
    def missing_value_summary(dataframe):
        nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
    
        summary_data = []
    
        for column in nan_columns:

            nan_number = dataframe[column].isna().sum()
    
            nan_percentage = (nan_number / len(dataframe)) * 100

            unique_values = dataframe[column].nunique()
        
            summary_data.append({
                'Unique Values': unique_values,
                'NaN Values': nan_number,
                'Percentage NaN': nan_percentage
            })
    
        summary = pd.DataFrame(summary_data, index=nan_columns)
    
        return summary

    def fix_zip_code(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))[:5]
        if isinstance(x, str) and x.isnumeric():
            return str(int(x))[:5]
        try:
            return(str(int(x))[:5])
        except:
            warnings.warn("x cannot be turned into a string of 5 characters", UserWarning)

    def calculate_days_until_reference(df, reference_date='2023-12-25'):
        reference_date = pd.to_datetime(reference_date)
        date_columns = ['Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']

        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            df[col] = (reference_date - df[col]).dt.days

        return df

    def handle_outliers(df):
        df_clean = df.copy()
        modifications = {}

        # Track modifications for each column
        for col in df_clean.columns:
            modifications[col] = {
                'original_count': len(df_clean),
                'modified_count': 0,
                'lower_bound': None,
                'upper_bound': None
            }

        # Date fields
        date_cols = ['C-2 Date', 'C-3 Date', 'First Hearing Date']
        for col in date_cols:
            lower_bound = df_clean[col].quantile(0.001)
            upper_bound = df_clean[col].quantile(0.999)
            lower_mask = df_clean[col] < lower_bound
            upper_mask = df_clean[col] > upper_bound

            df_clean.loc[lower_mask, col] = lower_bound
            df_clean.loc[upper_mask, col] = upper_bound

            modifications[col].update({
                'modified_count': (lower_mask | upper_mask).sum(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })

        # Birth Year
        birth_lower = df_clean['Birth Year'].quantile(0.001)
        birth_upper = df_clean['Birth Year'].quantile(0.999)
        birth_lower_mask = df_clean['Birth Year'] < birth_lower
        birth_upper_mask = df_clean['Birth Year'] > birth_upper

        df_clean.loc[birth_lower_mask, 'Birth Year'] = birth_lower
        df_clean.loc[birth_upper_mask, 'Birth Year'] = birth_upper

        modifications['Birth Year'].update({
            'modified_count': (birth_lower_mask | birth_upper_mask).sum(),
            'lower_bound': birth_lower,
            'upper_bound': birth_upper
        })

        # IME-4 Count
        ime_lower = df_clean['IME-4 Count'].quantile(0.025)
        ime_upper = df_clean['IME-4 Count'].quantile(0.975)
        ime_lower_mask = df_clean['IME-4 Count'] < ime_lower
        ime_upper_mask = df_clean['IME-4 Count'] > ime_upper

        df_clean.loc[ime_lower_mask, 'IME-4 Count'] = ime_lower
        df_clean.loc[ime_upper_mask, 'IME-4 Count'] = ime_upper

        modifications['IME-4 Count'].update({
            'modified_count': (ime_lower_mask | ime_upper_mask).sum(),
            'lower_bound': ime_lower,
            'upper_bound': ime_upper
        })

            # Average Weekly Wage
        wage_lower = df_clean['Average Weekly Wage'].quantile(0.0025)
        wage_upper = df_clean['Average Weekly Wage'].quantile(0.9975)
        wage_lower_mask = df_clean['Average Weekly Wage'] < wage_lower
        wage_upper_mask = df_clean['Average Weekly Wage'] > wage_upper

        df_clean.loc[wage_lower_mask, 'Average Weekly Wage'] = wage_lower
        df_clean.loc[wage_upper_mask, 'Average Weekly Wage'] = wage_upper

        modifications['Average Weekly Wage'].update({
            'modified_count': (wage_lower_mask | wage_upper_mask).sum(),
            'lower_bound': wage_lower,
            'upper_bound': wage_upper
        })

        return df_clean, modifications

    def validate_input_dataframe(df: Union[pd.DataFrame, Tuple]) -> pd.DataFrame:
            """
            Validate and extract DataFrame from input.

            Parameters:
                df: Input data structure (DataFrame or Tuple)

            Returns:
            pd.DataFrame: Validated DataFrame
            """
            if isinstance(df, pd.DataFrame):
                return df
            elif isinstance(df, tuple) and len(df) == 2:
                if isinstance(df[0], pd.DataFrame):
                    print("Input is a tuple. Extracting DataFrame.")
                    return df[0]
                else:
                    raise TypeError("Tuple does not contain a DataFrame.")
            else:
                raise TypeError("Input must be a pandas DataFrame or a tuple containing a DataFrame.")

    def identify_binary_columns(df: pd.DataFrame) -> list:
        """
        Identify columns in the DataFrame that are binary.

        Parameters:
            df: DataFrame to analyze

        Returns:
            list: List of binary column names
        """
        binary_columns = []
        for col in df.columns:
            if df[col].nunique() == 2:
                binary_columns.append(col)
        return binary_columns

    def scale_features(df_input: Union[pd.DataFrame, Tuple]) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale continuous features using StandardScaler.

        Parameters:
            df_input: Input DataFrame or tuple containing DataFrame

        Returns:
            tuple: (scaled_dataframe, scalers_dictionary)
        """
        # Validate input
        df = validate_input_dataframe(df_input)

        # Identify binary columns
        binary_cols = identify_binary_columns(df)

        # Get continuous columns
        continuous_cols = [col for col in df.columns if col not in binary_cols]

        # Initialize scaler
        scaler = StandardScaler()

        # Scale continuous features
        df_scaled = df.copy()
        df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])

        # Return scaled DataFrame and scaler
        scalers = {'continuous_scaler': scaler, 'binary_columns': binary_cols}
        return df_scaled, scalers

    def apply_scaling(df_input: Union[pd.DataFrame, Tuple], scalers: Dict) -> pd.DataFrame:
            """
            Apply scaling to new data using pre-fitted scalers.
    
            Parameters:
                df_input: Input DataFrame or tuple containing DataFrame
                scalers: Dictionary containing the scaler and binary columns

            Returns:
                pd.DataFrame: Scaled DataFrame
            """
            # Validate input
            df = validate_input_dataframe(df_input)

            # Get scaler and binary columns from scalers dictionary
            continuous_scaler = scalers['continuous_scaler']
            binary_cols = scalers['binary_columns']

            # Get continuous columns present in the new DataFrame
            continuous_cols = [col for col in df.columns if col not in binary_cols]

            # Apply scaling to continuous features
            df_scaled = df.copy()
            df_scaled[continuous_cols] = continuous_scaler.transform(df[continuous_cols])

            return df_scaled

            # Execution pipeline with error handling
            try:
                # Initial scaling on training data
                train_scaled, scalers = scale_features(train_clean)
        
                # Apply scaling to validation and test sets
                val_scaled = apply_scaling(val_clean, scalers)
                test_scaled = apply_scaling(test_clean, scalers)
        
            except TypeError as e:
                print(f"TypeError occurred: {e}")
            except ValueError as e:
                print(f"ValueError occurred: {e}")

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

   def engineer_features(df):
   # WCIO PCA
      wcio_features = ['WCIO Nature of Injury Code', 'WCIO Part Of Body Code', 'WCIO Cause of Injury Code']
      pca = PCA(n_components=2)
      wcio_pca = pca.fit_transform(StandardScaler().fit_transform(df[wcio_features]))
      df['wcio_pca1'] = wcio_pca[:, 0]
      df['wcio_pca2'] = wcio_pca[:, 1]

      # Region clustering
      df['region_cluster'] = df['Medical Fee Region'].astype(str) + '_' + df['Zip Code'].astype(str) + df['County of Injury'].astype(str)

      # Market indicator based on WCIO frequencies
      high_risk_nature = [20658, 12456, 28947]
      high_risk_body = [11229, 8780, 3028]
      high_risk_cause = [20096, 12441, 18015, 340, 1989]

      # Add high risk indicators
      df['high_risk_nature'] = df['WCIO Nature of Injury Code'].isin(high_risk_nature).astype(int)
      df['high_risk_body'] = df['WCIO Part Of Body Code'].isin(high_risk_body).astype(int)
      df['high_risk_cause'] = df['WCIO Cause of Injury Code'].isin(high_risk_cause).astype(int)

      df['market_indicator'] = ((df['high_risk_nature']) |
                             (df['high_risk_body']) |
                             (df['high_risk_cause'])).astype(int)
   
      # Drop redundant columns
      cols_to_drop = [] #'Birth Year', 'COVID-19 Indicator_Y' + wcio_features

      return df.drop(columns=cols_to_drop)
       
   def missing_value_summary(dataframe):
       nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
    
       summary_data = []
    
       for column in nan_columns:

           nan_number = dataframe[column].isna().sum()
   
           nan_percentage = (nan_number / len(dataframe)) * 100

           unique_values = dataframe[column].nunique()
        
           summary_data.append({
               'Unique Values': unique_values,
               'NaN Values': nan_number,
               'Percentage NaN': nan_percentage
           })
    
       summary = pd.DataFrame(summary_data, index=nan_columns)
    
       return summary

   def fix_zip_code(x):
       if isinstance(x, float) and x.is_integer():
           return str(int(x))[:5]
       if isinstance(x, str) and x.isnumeric():
           return str(int(x))[:5]
       try:
           return(str(int(x))[:5])
       except:
           warnings.warn("x cannot be turned into a string of 5 characters", UserWarning)

   def calculate_days_until_reference(df, reference_date='2023-12-25'):
       reference_date = pd.to_datetime(reference_date)
       date_columns = ['Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']

       for col in date_columns:
           df[col] = pd.to_datetime(df[col])
           df[col] = (reference_date - df[col]).dt.days

       return df

   def handle_outliers(df):
       df_clean = df.copy()
       modifications = {}

       # Track modifications for each column
       for col in df_clean.columns:
           modifications[col] = {
               'original_count': len(df_clean),
               'modified_count': 0,
               'lower_bound': None,
               'upper_bound': None
           }

       # Date fields
       date_cols = ['C-2 Date', 'C-3 Date', 'First Hearing Date']
       for col in date_cols:
           lower_bound = df_clean[col].quantile(0.001)
           upper_bound = df_clean[col].quantile(0.999)
           lower_mask = df_clean[col] < lower_bound
           upper_mask = df_clean[col] > upper_bound

           df_clean.loc[lower_mask, col] = lower_bound
           df_clean.loc[upper_mask, col] = upper_bound

           modifications[col].update({
               'modified_count': (lower_mask | upper_mask).sum(),
               'lower_bound': lower_bound,
               'upper_bound': upper_bound
           })

       # Birth Year
       birth_lower = df_clean['Birth Year'].quantile(0.001)
       birth_upper = df_clean['Birth Year'].quantile(0.999)
       birth_lower_mask = df_clean['Birth Year'] < birth_lower
       birth_upper_mask = df_clean['Birth Year'] > birth_upper

       df_clean.loc[birth_lower_mask, 'Birth Year'] = birth_lower
       df_clean.loc[birth_upper_mask, 'Birth Year'] = birth_upper

       modifications['Birth Year'].update({
           'modified_count': (birth_lower_mask | birth_upper_mask).sum(),
           'lower_bound': birth_lower,
           'upper_bound': birth_upper
       })

       # IME-4 Count
       ime_lower = df_clean['IME-4 Count'].quantile(0.025)
       ime_upper = df_clean['IME-4 Count'].quantile(0.975)
       ime_lower_mask = df_clean['IME-4 Count'] < ime_lower
       ime_upper_mask = df_clean['IME-4 Count'] > ime_upper

       df_clean.loc[ime_lower_mask, 'IME-4 Count'] = ime_lower
       df_clean.loc[ime_upper_mask, 'IME-4 Count'] = ime_upper

       modifications['IME-4 Count'].update({
           'modified_count': (ime_lower_mask | ime_upper_mask).sum(),
           'lower_bound': ime_lower,
           'upper_bound': ime_upper
       })

           # Average Weekly Wage
       wage_lower = df_clean['Average Weekly Wage'].quantile(0.0025)
       wage_upper = df_clean['Average Weekly Wage'].quantile(0.9975)
       wage_lower_mask = df_clean['Average Weekly Wage'] < wage_lower
       wage_upper_mask = df_clean['Average Weekly Wage'] > wage_upper

       df_clean.loc[wage_lower_mask, 'Average Weekly Wage'] = wage_lower
       df_clean.loc[wage_upper_mask, 'Average Weekly Wage'] = wage_upper

       modifications['Average Weekly Wage'].update({
           'modified_count': (wage_lower_mask | wage_upper_mask).sum(),
           'lower_bound': wage_lower,
           'upper_bound': wage_upper
       })

       return df_clean, modifications

   def validate_input_dataframe(df: Union[pd.DataFrame, Tuple]) -> pd.DataFrame:
           """
           Validate and extract DataFrame from input.

           Parameters:
               df: Input data structure (DataFrame or Tuple)

           Returns:
           pd.DataFrame: Validated DataFrame
           """
           if isinstance(df, pd.DataFrame):
               return df
           elif isinstance(df, tuple) and len(df) == 2:
               if isinstance(df[0], pd.DataFrame):
                   print("Input is a tuple. Extracting DataFrame.")
                   return df[0]
               else:
                   raise TypeError("Tuple does not contain a DataFrame.")
           else:
               raise TypeError("Input must be a pandas DataFrame or a tuple containing a DataFrame.")

   def identify_binary_columns(df: pd.DataFrame) -> list:
       """
       Identify columns in the DataFrame that are binary.

       Parameters:
           df: DataFrame to analyze

       Returns:
           list: List of binary column names
       """
       binary_columns = []
       for col in df.columns:
           if df[col].nunique() == 2:
               binary_columns.append(col)
       return binary_columns

   def scale_features(df_input: Union[pd.DataFrame, Tuple]) -> Tuple[pd.DataFrame, Dict]:
       """
       Scale continuous features using StandardScaler.

       Parameters:
           df_input: Input DataFrame or tuple containing DataFrame

       Returns:
           tuple: (scaled_dataframe, scalers_dictionary)
       """
       # Validate input
       df = validate_input_dataframe(df_input)

       # Identify binary columns
       binary_cols = identify_binary_columns(df)

       # Get continuous columns
       continuous_cols = [col for col in df.columns if col not in binary_cols]

       # Initialize scaler
       scaler = StandardScaler()

       # Scale continuous features
       df_scaled = df.copy()
       df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])

       # Return scaled DataFrame and scaler
       scalers = {'continuous_scaler': scaler, 'binary_columns': binary_cols}
       return df_scaled, scalers

   def apply_scaling(df_input: Union[pd.DataFrame, Tuple], scalers: Dict) -> pd.DataFrame:
           """
           Apply scaling to new data using pre-fitted scalers.
    
           Parameters:
               df_input: Input DataFrame or tuple containing DataFrame
               scalers: Dictionary containing the scaler and binary columns

           Returns:
               pd.DataFrame: Scaled DataFrame
           """
           # Validate input
           df = validate_input_dataframe(df_input)

           # Get scaler and binary columns from scalers dictionary
           continuous_scaler = scalers['continuous_scaler']
           binary_cols = scalers['binary_columns']

           # Get continuous columns present in the new DataFrame
           continuous_cols = [col for col in df.columns if col not in binary_cols]

           # Apply scaling to continuous features
           df_scaled = df.copy()
           df_scaled[continuous_cols] = continuous_scaler.transform(df[continuous_cols])

           return df_scaled

           # Execution pipeline with error handling
           try:
               # Initial scaling on training data
               train_scaled, scalers = scale_features(train_clean)
        
               # Apply scaling to validation and test sets
               val_scaled = apply_scaling(val_clean, scalers)
               test_scaled = apply_scaling(test_clean, scalers)
        
           except TypeError as e:
               print(f"TypeError occurred: {e}")
           except ValueError as e:
               print(f"ValueError occurred: {e}")



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
    test

    
    train = pd.read_csv('train_data_mok.csv')
    
    train = train.drop(columns=['Industry Code Description'])
    test = test.drop(columns=['Industry Code Description'])
    claimMode = train['Claim Identifier'].mode()
    claimDateModeIndex = list(train[train['Claim Identifier'] == claimMode[0]].index)
    claimDateModeIndex
    train.drop(claimDateModeIndex, inplace=True)

    train.set_index('Claim Identifier', inplace=True)
    test.set_index('Claim Identifier', inplace=True)

    train.dropna(subset=['Claim Injury Type'], inplace=True)

    train.drop_duplicates(keep='first', inplace=True)

    train.drop('OIICS Nature of Injury Description', axis = 1, inplace = True)

    test.drop('OIICS Nature of Injury Description', axis = 1, inplace = True)

    train.drop('WCB Decision', axis = 1, inplace = True)

    train['Birth Year'] = train['Birth Year'].replace(0, np.nan)

    

    train['IME-4 Count'] = train['IME-4 Count'].fillna(0)
    test['IME-4 Count'] = test['IME-4 Count'].fillna(0)

    train["Industry Code"] = train["Industry Code"].fillna(0)
    test["Industry Code"] = test["Industry Code"].fillna(0)

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


  
    train = train.drop(columns=['WCIO Nature of Injury Description'])
    test = test.drop(columns=['WCIO Nature of Injury Description'])

    train = train.drop(columns=['WCIO Cause of Injury Description'])
    test = test.drop(columns=['WCIO Cause of Injury Description'])

    train = train.drop(columns=['WCIO Part Of Body Description'])
    test = test.drop(columns=['WCIO Part Of Body Description'])

    x = train.drop(columns= 'Claim Injury Type')
    y = train['Claim Injury Type']

    train_set, val_set, y_train, y_val = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y
                                                   )

    train_set['Accident Date'] = pd.to_datetime(train_set['Accident Date'])
    val_set['Accident Date'] = pd.to_datetime(val_set['Accident Date'])
    test['Accident Date'] = pd.to_datetime(test['Accident Date'])

    train_set['Accident Year'] = train_set['Accident Date'].apply(lambda x: x.year)
    val_set['Accident Year'] = val_set['Accident Date'].apply(lambda x: x.year)
    test['Accident Year'] = test['Accident Date'].apply(lambda x: x.year)
    
    train_set['Accident Month'] = train_set['Accident Date'].apply(lambda x: x.month)
    val_set['Accident Month'] = val_set['Accident Date'].apply(lambda x: x.month)
    test['Accident Month'] = test['Accident Date'].apply(lambda x: x.month)

    train_set['Accident Day'] = train_set['Accident Date'].apply(lambda x: x.day)
    val_set['Accident Day'] = val_set['Accident Date'].apply(lambda x: x.day)
    test['Accident Day'] = test['Accident Date'].apply(lambda x: x.day)

    train_set.apply(lambda row: row['Accident Year'] - row['Birth Year'] == row['Age at Injury'], axis=1).value_counts()

    train_set.loc[train_set['Birth Year'].isna() & train_set['Accident Year'].notna() & train_set['Age at Injury'].notna(), 'Birth Year'] = (
    train_set['Accident Year'] - train_set['Age at Injury'])

    val_set.loc[val_set['Birth Year'].isna() & val_set['Accident Year'].notna() & val_set['Age at Injury'].notna(), 'Birth Year'] = (
    val_set['Accident Year'] - val_set['Age at Injury'])

    train_set.loc[train_set["Birth Year"] == train_set["Accident Year"], "Birth Year"] = np.nan
    val_set.loc[val_set["Birth Year"] == val_set["Accident Year"], "Birth Year"] = np.nan

    train_set['Age at Injury'] = train_set['Accident Year'] - train_set['Birth Year']
    val_set['Age at Injury'] = val_set['Accident Year'] - val_set['Birth Year']

   


    train_set['Zip Code'] = train_set['Zip Code'].apply(fix_zip_code)
    val_set['Zip Code'] = val_set['Zip Code'].apply(fix_zip_code)

    train_set['Assembly Date'] = pd.to_datetime(train_set['Assembly Date'])
    train_set['Accident Date'] = pd.to_datetime(train_set['Accident Date'])
    train_set['Days_to_Assembly'] = (train_set['Assembly Date'] - train_set['Accident Date']).dt.days

    val_set['Assembly Date'] = pd.to_datetime(val_set['Assembly Date'])
    val_set['Accident Date'] = pd.to_datetime(val_set['Accident Date'])
    val_set['Days_to_Assembly'] = (val_set['Assembly Date'] - val_set['Accident Date']).dt.days

    test['Assembly Date'] = pd.to_datetime(test['Assembly Date'])
    test['Accident Date'] = pd.to_datetime(test['Accident Date'])
    test['Days_to_Assembly'] = (test['Assembly Date'] - test['Accident Date']).dt.days

    train_set['Days_to_Assembly'] = (train_set['Assembly Date'] - train_set['Accident Date']).dt.days
    val_set['Days_to_Assembly'] = (val_set['Assembly Date'] - val_set['Accident Date']).dt.days
    test['Days_to_Assembly'] = (test['Assembly Date'] - test['Accident Date']).dt.days

    train_set['Days_to_Assembly'] = train_set['Days_to_Assembly'].apply((lambda x: 1 if x < 365 else 0))
    val_set['Days_to_Assembly'] = val_set['Days_to_Assembly'].apply((lambda x: 1 if x < 365 else 0))
    test['Days_to_Assembly'] = test['Days_to_Assembly'].apply((lambda x: 1 if x < 365 else 0))

    train_set['Under_20'] = (train_set['Age at Injury'] < 20).astype(int)
    train_set['Age_21_40'] = ((train_set['Age at Injury'] >= 21) & (train_set['Age at Injury'] <= 40)).astype(int)
    train_set['Age_41_65'] = ((train_set['Age at Injury'] >= 41) & (train_set['Age at Injury'] <= 65)).astype(int)
    train_set['Above_65'] = (train_set['Age at Injury'] > 65).astype(int)

    val_set['Under_20'] = (val_set['Age at Injury'] < 20).astype(int)
    val_set['Age_21_40'] = ((val_set['Age at Injury'] >= 21) & (val_set['Age at Injury'] <= 40)).astype(int)
    val_set['Age_41_65'] = ((val_set['Age at Injury'] >= 41) & (val_set['Age at Injury'] <= 65)).astype(int)
    val_set['Above_65'] = (val_set['Age at Injury'] > 65).astype(int)

    test['Under_20'] = (test['Age at Injury'] < 20).astype(int)
    test['Age_21_40'] = ((test['Age at Injury'] >= 21) & (test['Age at Injury'] <= 40)).astype(int)
    test['Age_41_65'] = ((test['Age at Injury'] >= 41) & (test['Age at Injury'] <= 65)).astype(int)
    test['Above_65'] = (test['Age at Injury'] > 65).astype(int)

    train_set = train_set.drop(columns = ['Accident Date', 'Age at Injury'])
    val_set = val_set.drop(columns = ['Accident Date', 'Age at Injury'])
    test = test.drop(columns = ['Accident Date', 'Age at Injury'])

    frequency_map_dependents = train_set['Number of Dependents'].value_counts(normalize=True)

    train_set['Number of Dependents'] = train_set['Number of Dependents'].map(frequency_map_dependents)
    val_set['Number of Dependents'] = val_set['Number of Dependents'].map(frequency_map_dependents)
    test['Number of Dependents'] = test['Number of Dependents'].map(frequency_map_dependents)

    train_set['Accident Month Sin'] = np.sin(2 * np.pi * train_set['Accident Month'] / 12)
    train_set['Accident Month Cos'] = np.cos(2 * np.pi * train_set['Accident Month'] / 12)

    val_set['Accident Month Sin'] = np.sin(2 * np.pi * val_set['Accident Month'] / 12)
    val_set['Accident Month Cos'] = np.cos(2 * np.pi * val_set['Accident Month'] / 12)

    test['Accident Month Sin'] = np.sin(2 * np.pi * test['Accident Month'] / 12)
    test['Accident Month Cos'] = np.cos(2 * np.pi * test['Accident Month'] / 12)

   
    train_set['Accident Day Sin'] = np.sin(2 * np.pi * train_set['Accident Day'] / 31)
    train_set['Accident Day Cos'] = np.cos(2 * np.pi * train_set['Accident Day'] / 31)

    val_set['Accident Day Sin'] = np.sin(2 * np.pi * val_set['Accident Day'] / 31)
    val_set['Accident Day Cos'] = np.cos(2 * np.pi * val_set['Accident Day'] / 31)

    test['Accident Day Sin'] = np.sin(2 * np.pi * test['Accident Day'] / 31)
    test['Accident Day Cos'] = np.cos(2 * np.pi * test['Accident Day'] / 31)


    train_set["Received_C2"] = train_set["C-2 Date"].apply(lambda x: 0 if pd.isna(x) else 1)
    val_set["Received_C2"] = val_set["C-2 Date"].apply(lambda x: 0 if pd.isna(x) else 1)
    test["Received_C2"] = test["C-2 Date"].apply(lambda x: 0 if pd.isna(x) else 1)

    train_set["Received_C3"] = train_set["C-3 Date"].apply(lambda x: 0 if pd.isna(x) else 1)
    val_set["Received_C3"] = val_set["C-3 Date"].apply(lambda x: 0 if pd.isna(x) else 1)
    test["Received_C3"] = test["C-3 Date"].apply(lambda x: 0 if pd.isna(x) else 1)

    train_set["Hearing_held"] = train_set["First Hearing Date"].apply(lambda x: 0 if pd.isna(x) else 1)
    val_set["Hearing_held"] = val_set["First Hearing Date"].apply(lambda x: 0 if pd.isna(x) else 1)
    test["Hearing_held"] = test["First Hearing Date"].apply(lambda x: 0 if pd.isna(x) else 1)

    train_set = pd.get_dummies(train_set, columns=['Attorney/Representative', 'COVID-19 Indicator','Alternative Dispute Resolution', 'Gender'], drop_first=True, dtype=int)
    val_set = pd.get_dummies(val_set, columns=['Attorney/Representative', 'COVID-19 Indicator','Alternative Dispute Resolution', 'Gender'], drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=['Attorney/Representative', 'COVID-19 Indicator','Alternative Dispute Resolution', 'Gender'], drop_first=True, dtype=int)

    train_set.drop(columns='Gender_U', inplace=True, errors='ignore')
    val_set.drop(columns='Gender_U', inplace=True, errors='ignore')
    test.drop(columns='Gender_U', inplace=True, errors='ignore')
    train_set.drop(columns='Gender_X', inplace=True, errors='ignore')
    val_set.drop(columns='Gender_X', inplace=True, errors='ignore')
    test.drop(columns='Gender_X', inplace=True, errors='ignore')

    frequency_map_ic = train_set['Industry Code'].value_counts(normalize=False)

    train_set['Industry Code'] = train_set['Industry Code'].map(frequency_map_ic)
    val_set['Industry Code'] = val_set['Industry Code'].map(frequency_map_ic)
    test['Industry Code'] = test['Industry Code'].map(frequency_map_ic)
    
    frequency_map_wcio_ic = train_set['WCIO Cause of Injury Code'].value_counts(normalize=False)

    train_set['WCIO Cause of Injury Code'] = train_set['WCIO Cause of Injury Code'].map(frequency_map_wcio_ic)
    val_set['WCIO Cause of Injury Code'] = val_set['WCIO Cause of Injury Code'].map(frequency_map_wcio_ic)
    test['WCIO Cause of Injury Code'] = test['WCIO Cause of Injury Code'].map(frequency_map_wcio_ic)

    frequency_map_wcio_nic = train_set['WCIO Nature of Injury Code'].value_counts(normalize=False)

    train_set['WCIO Nature of Injury Code'] = train_set['WCIO Nature of Injury Code'].map(frequency_map_wcio_nic)
    val_set['WCIO Nature of Injury Code'] = val_set['WCIO Nature of Injury Code'].map(frequency_map_wcio_nic)
    test['WCIO Nature of Injury Code'] = test['WCIO Nature of Injury Code'].map(frequency_map_wcio_nic)

    frequency_map_wcio_pbc = train_set['WCIO Part Of Body Code'].value_counts(normalize=False)

    train_set['WCIO Part Of Body Code'] = train_set['WCIO Part Of Body Code'].map(frequency_map_wcio_pbc)
    val_set['WCIO Part Of Body Code'] = val_set['WCIO Part Of Body Code'].map(frequency_map_wcio_pbc)
    test['WCIO Part Of Body Code'] = test['WCIO Part Of Body Code'].map(frequency_map_wcio_pbc)

    frequency_map_mfr = train_set['Medical Fee Region'].value_counts(normalize=False)

    train_set['Medical Fee Region'] = train_set['Medical Fee Region'].map(frequency_map_mfr)
    val_set['Medical Fee Region'] = val_set['Medical Fee Region'].map(frequency_map_mfr)
    test['Medical Fee Region'] = test['Medical Fee Region'].map(frequency_map_mfr)

    frequency_map_ct = train_set['Carrier Type'].value_counts(normalize=False)
    frequency_map_cn = train_set['Carrier Name'].value_counts(normalize=False)
    frequency_map_coi = train_set['County of Injury'].value_counts(normalize=False)
    frequency_map_dn = train_set['District Name'].value_counts(normalize=False)

    train_set['Carrier Type'] = train_set['Carrier Type'].map(frequency_map_ct)
    val_set['Carrier Type'] = val_set['Carrier Type'].map(frequency_map_ct)
    test['Carrier Type'] = test['Carrier Type'].map(frequency_map_ct)

    train_set['Carrier Name'] = train_set['Carrier Name'].map(frequency_map_cn)
    val_set['Carrier Name'] = val_set['Carrier Name'].map(frequency_map_cn)
    test['Carrier Name'] = test['Carrier Name'].map(frequency_map_cn)

    train_set['County of Injury'] = train_set['County of Injury'].map(frequency_map_coi)
    val_set['County of Injury'] = val_set['County of Injury'].map(frequency_map_coi)
    test['County of Injury'] = test['County of Injury'].map(frequency_map_coi)

    train_set['District Name'] = train_set['District Name'].map(frequency_map_dn)
    val_set['District Name'] = val_set['District Name'].map(frequency_map_dn)
    test['District Name'] = test['District Name'].map(frequency_map_dn)
    
    col_fill_median = ['Average Weekly Wage', 'Birth Year', 'Accident Year']

    for col in col_fill_median:
        median_value = train_set[col].median()

        train_set[col].fillna(median_value, inplace=True)
        val_set[col].fillna(median_value, inplace=True)
        test[col].fillna(median_value, inplace=True)

    col_fill_mode = ['C-2 Date', 'C-3 Date', 'First Hearing Date', 'Zip Code']

    for col in col_fill_mode:
        mode_value = train_set[col].mode()[0]

        train_set[col].fillna(mode_value, inplace=True)
        val_set[col].fillna(mode_value, inplace=True)
        test[col].fillna(mode_value, inplace=True)

    
    

    train_set = calculate_days_until_reference(train_set)
    val_set = calculate_days_until_reference(val_set)
    test = calculate_days_until_reference(test)

    train_numerical = train_set.select_dtypes(include='number')
    train_categorical = train_set.select_dtypes(include='object')



    train_clean = handle_outliers(train_set)
    val_clean = handle_outliers(val_set)
    test_clean = handle_outliers(test)

    # Columns to impute with median
    cols_to_impute = ['Accident Month', 'Accident Day', 'Accident Month Sin',
                     'Accident Month Cos', 'Accident Day Sin', 'Accident Day Cos']

    # Impute with median
    for col in cols_to_impute:
        median = train_set[col].median()
        train_set[col].fillna(median, inplace=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Extract first DataFrame from tuple
    train_clean = train_clean[0] if isinstance(train_clean, tuple) else train_clean
    test_clean = test_clean[0] if isinstance(test_clean, tuple) else test_clean
    val_clean = val_clean[0] if isinstance(val_clean, tuple) else val_clean

    # Apply feature engineering
    train_engineered = engineer_features(train_clean)
    test_engineered = engineer_features(test_clean)
    val_engineered = engineer_features(val_clean)

    train_clean = train_clean[0] if isinstance(train_clean, tuple) else train_clean
    test_clean = test_clean[0] if isinstance(test_clean, tuple) else test_clean
    val_clean = val_clean[0] if isinstance(val_clean, tuple) else val_clean

    # Apply feature engineering
    train_engineered = engineer_features(train_clean)
    test_engineered = engineer_features(test_clean)
    val_engineered = engineer_features(val_clean)

    frequency_map_region_cluster = train_engineered['region_cluster'].value_counts(normalize=False)
    train_engineered['region_cluster'] = train_engineered['region_cluster'].map(frequency_map_region_cluster)
    test_engineered['region_cluster'] = test_engineered['region_cluster'].map(frequency_map_region_cluster)
    val_engineered['region_cluster'] = val_engineered['region_cluster'].map(frequency_map_region_cluster)

    train_engineered = train_engineered.drop(columns=['Medical Fee Region', 'Zip Code'])
    val_engineered = val_engineered.drop(columns=['Medical Fee Region', 'Zip Code'])
    test_engineered = test_engineered.drop(columns=['Medical Fee Region', 'Zip Code'])






    # Initialize with quantum precision
    train_scaled, scalers = scale_features(train_engineered)

    # Validate with Heisenberg certainty
    val_scaled = apply_scaling(val_engineered, scalers)
    test_scaled = apply_scaling(test_engineered, scalers)

    scaler = StandardScaler()

    columns_to_scale = ['Carrier Name', 'Carrier Type', 'District Name', 'Industry Code', 'region_cluster']
    train_scaled[columns_to_scale] = scaler.fit_transform(train_scaled[columns_to_scale])
    val_scaled[columns_to_scale] = scaler.transform(val_scaled[columns_to_scale])  # Use the same scaler fitted on train_set
    test_scaled[columns_to_scale] = scaler.transform(test_scaled[columns_to_scale])  # Use the same scaler fitted on train_set

    val_full = pd.merge(y, val_scaled, on='Claim Identifier', how='right')

    train_full = pd.merge(y, train_scaled, on='Claim Identifier', how='right')

    train_scaled.drop(columns=['Alternative Dispute Resolution_U'], errors='ignore')
test_scaled
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
            model.fit(train_scaled, y)
            prediction = model.predict(test_scaled)
            st.success(f'Prediction: {prediction[0]}')
    
