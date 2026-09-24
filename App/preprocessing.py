from sklearn import set_config
import pandas as pd

set_config(transform_output="pandas")

def cast_taxi_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Float conversions
    float_cols = ['trip_distance', 'duration']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')
            
    # 2. Integer conversions (Int8 handles nullable integers)
    if 'RatecodeID' in df.columns:
        df['RatecodeID'] = df['RatecodeID'].astype('Int8')
        
    int8_cols = ['day_of_the_week', 'pick_up_month']
    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].astype('int8')
            
    # 3. Boolean conversions
    bool_cols = [
        'is_weekend', 'is_rush_hour', 'is_late_night',
        'is_airport_pickup', 'is_airport_dropoff', 'SameZone'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('bool')
            
    # 4. Categorical conversions (XGBoost can consume these natively)
    cat_cols = ['boroughPickUpPoint', 'boroughDropOffPoint']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df
