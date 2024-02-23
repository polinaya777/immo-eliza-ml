import numpy as np
import pandas as pd
from scipy.stats import zscore

def manual_preprocessing(file_path: str, save_path: str) -> None:
    """Preprocesses the data for ML training and prediction."""

    # _____ Load the data _____ 
    data = pd.read_csv(file_path)
    data_origin = data.copy()


    # _____ Drop duplicates and some missing values _____ 
    # Drop rows with missing 'region' values
    data = data[data['region'] != 'MISSING']
    data = data.reset_index(drop=True)

    # Drop rows with missing 'total_area_sqm' values
    data = data.drop(data[data['total_area_sqm'].isna()].index)
    data = data.reset_index(drop=True)

    # Drop duplicates (except for the 'id' column)
    data = data.drop_duplicates(subset=data.columns[1:], keep='first')
    data = data.reset_index(drop=True)


    # _____ Manage outliers _____ 
    # Remove outliers for numerical columns withing each 'property_type' group

    # Define a custom function to filter out outliers based on z-score
    def remove_outliers(df):
        # Calculate z-scores for numerical columns
        numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
        z_scores = np.abs(df[numerical_cols].apply(zscore))    
        # Identify rows with any numerical column having a z-score > 3
        outlier_mask = (z_scores > 3).any(axis=1)    
        # Return filtered DataFrame excluding identified outliers
        return df[~outlier_mask]

    # Apply the custom function to each property_type category separately
    data = data.groupby('property_type').apply(remove_outliers).reset_index(drop=True)
    data = data.reset_index(drop=True)


    # _____ Manage inconsistent values _____
    # Drop inconsistent 'terrace_sqm' and 'garden_sqm' values
    data = data.drop(data[(data['terrace_sqm'] > data['surface_land_sqm']) | 
                          (data['garden_sqm'] > data['surface_land_sqm'])].index)
    data = data.reset_index(drop=True)
    
    # Drop inconsistent 'primary_energy_consumption_sqm' and 'epc' values
    data = data.drop(data[(data['primary_energy_consumption_sqm'] == 0) & 
                          (data['epc'] == 'MISSING')].index)
    data = data.reset_index(drop=True)
    data = data.drop(data[(data['primary_energy_consumption_sqm'].isna()) & 
                          (data['epc'] == 'MISSING')].index)
    data = data.reset_index(drop=True)
    data = data.drop(data[(data["epc"] == 'A++') & (data['region'] == 'Brussels-Capital')].index)
    data = data.reset_index(drop=True)
    data = data[data['primary_energy_consumption_sqm'] <= 1200]
    data = data.reset_index(drop=True)

    # Drop negative 'primary_energy_consumption_sqm' values if 'epc' is not 'A++', 'A+', 'A'
    data = data.drop(data[(data['primary_energy_consumption_sqm'] < 0) & 
                          ~data['epc'].isin(['A++', 'A+', 'A'])].index)
    data = data.reset_index(drop=True)

    # Fill missing 'surface_land_sqm' values for 'apartment' property type
    data.loc[(data['property_type'] == 'apartment') & (data['surface_land_sqm'].isnull()), 
             'surface_land_sqm'] = 0
    
    # Filter 'epc' values depending on 'region'
    def filter_region_data(data, region, conditions):
        """Filter data based on specified region and conditions."""
        region_mask = data['region'].isin([region])
        condition_masks = []

        for min_sqm, max_sqm, epc_rating in conditions:
            if max_sqm is None:  # For the open-ended condition
                sqm_condition = (data['primary_energy_consumption_sqm'] > min_sqm)
            else:
                sqm_condition = ((data['primary_energy_consumption_sqm'] > min_sqm) & 
                                (data['primary_energy_consumption_sqm'] <= max_sqm))
            
            epc_condition = ~data['epc'].isin([epc_rating])
            condition_masks.append(sqm_condition & epc_condition)
        
        # Combine all conditions with the region condition
        final_mask = region_mask & pd.concat(condition_masks, axis=1).any(axis=1)
        filtered_indices = data[final_mask].index
        return data.drop(filtered_indices)

    # Define conditions for each region
    flanders_conditions = [
        (0, 100, 'A'),
        (100, 200, 'B'),
        (200, 300, 'C'),
        (300, 400, 'D'),
        (400, 500, 'E'),
        (500, None, 'F')  # Open-ended condition for > 500
        ]
    brussels_conditions = [
        (0, 45, 'A'),
        (45, 95, 'B'),
        (95, 150, 'C'),
        (150, 210, 'D'),
        (210, 280, 'E'),
        (280, 350, 'F'),
        (350, None, 'G')  # Open-ended condition for > 350
        ]
    wallonia_conditions = [
        (0, 45, 'A'),
        (45, 170, 'B'),
        (170, 260, 'C'),
        (260, 340, 'D'),
        (340, 425, 'E'),
        (425, 510, 'F'),
        (510, None, 'G')  # Open-ended condition for > 510
        ]

    # Apply the filter_region_data function to each region
    data = filter_region_data(data, 'Flanders', flanders_conditions)
    data = data.reset_index(drop=True)
    data = filter_region_data(data, 'Wallonia', wallonia_conditions)
    data = data.reset_index(drop=True)
    data = filter_region_data(data, 'Brussels-Capital', brussels_conditions)
    data = data.reset_index(drop=True)


    # Impute missing values in 'primary_energy_consumption_sqm' depending on 'epc' rating and 'region'
    # Replace zero values in 'primary_energy_consumption_sqm' with NaN only for specific EPC ratings
    data.loc[(data['primary_energy_consumption_sqm'] == 0) & (~data['epc'].isin(['A++', 'A+', 
                'A'])), 'primary_energy_consumption_sqm'] = np.nan

    # Calculate the mean 'primary_energy_consumption_sqm' for each 'region' and 'epc' group
    # and fill NaN values in 'primary_energy_consumption_sqm' with these means
    data['primary_energy_consumption_sqm'] = data.groupby(['region', 
        'epc'])['primary_energy_consumption_sqm'].transform(lambda x: x.fillna(x.mean()))

    # If there are still any NaN values in 'primary_energy_consumption_sqm' (e.g., all values in 
    # a group are NaN), fill them with a global mean or another placeholder value:
    global_mean = data['primary_energy_consumption_sqm'].mean()
    data['primary_energy_consumption_sqm'] = data['primary_energy_consumption_sqm'].fillna(global_mean)


    # Calculate the mean 'total_area_sqm' for each 'property_type' and 'nbr_bedrooms'
    # and fill NaN values in 'total_area_sqm' with these means
    data['total_area_sqm'] = data.groupby(['property_type', 
        'nbr_bedrooms'])['total_area_sqm'].transform(lambda x: x.fillna(x.mean()))
    
    # If there are still any NaN values in 'total_area_sqm' (e.g., all values in 
    # a group are NaN), fill them with a global mean or another placeholder value:
    global_mean = data['total_area_sqm'].mean()
    data['total_area_sqm'] = data['total_area_sqm'].fillna(global_mean)
    

    # _____ Save the preprocessed data _____
    data.to_csv(save_path, index=False)
    
    

    

