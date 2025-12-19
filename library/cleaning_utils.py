import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def remove_irrelevant_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns from the road accidents DataFrame.
    If a column does not exist, ignore the error and print its name.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with accident records.
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with specified columns dropped.
    """
    cols_to_drop = [
        "veh_id",
        "ind_year",
        "ind_age",
        "ind_secu2",
        "ind_location",
        "ind_action",
        "acc_metro",
        "acc_long",
        "acc_lat",
        "acc_date",
        "acc_department"
        
    ]
    
    missing_cols = [col for col in cols_to_drop if col not in df.columns]
    if missing_cols:
        print("⚠️ The following columns were not found and skipped:", missing_cols)
    
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns])


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def distinguish_cols(df: pd.DataFrame) -> dict:
    """
    Distinguish columns of the accident DataFrame into categories:
    categorical, ordinal, nominal, quantitative, and cyclic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    dict
        Dictionary with keys 'categorical', 'ordinal', 'nominal', 'quantitative', 'cyclic'
        and values as lists of columns present in df.
        Missing columns are ignored but reported.
    """
    cols_cat = [
        'acc_year','acc_municipality','acc_ambient_lightning','acc_urbanization_level',
        'acc_intersection','acc_atmosphere','acc_collision_type','ind_place','ind_cat',
        'ind_sex','ind_trip','ind_location','ind_action','ind_secu1','ind_secu2',
        'ind_age_group','loca_road_cat','loca_traffic_circul','loca_road_gradient',
        'loca_road_view','loca_road_surface_cond','loca_accident','loca_is_intersection',
        'veh_cat','veh_fixed_obstacle','veh_moving_obstacle','veh_impact','veh_maneuver',
        'veh_motor'
    ]

    cols_ordinal = [
        'acc_year','ind_age_group','loca_road_cat','loca_road_surface_cond'
    ]

    cols_nominal = [
        'acc_department','acc_municipality','acc_ambient_lightning','acc_urbanization_level',
        'acc_intersection','acc_atmosphere','acc_collision_type','ind_place','ind_cat',
        'ind_sex','ind_trip','ind_location','ind_action','ind_secu1','ind_secu2',
        'loca_traffic_circul','loca_road_gradient','loca_road_view','loca_accident',
        'loca_is_intersection','veh_cat','veh_fixed_obstacle','veh_moving_obstacle',
        'veh_impact','veh_maneuver','veh_motor'
    ]

    cols_quant = ['loca_road_lanes','loca_max_speed','loca_road_count']
    
    cols_cyclic = ['acc_month','acc_hour']

    cols_target_encoder = ['acc_municipality']

    cols_one_hot_encoder = [
        'acc_ambient_lightning','acc_urbanization_level','acc_intersection','acc_atmosphere','acc_collision_type',
        'ind_place','ind_cat','ind_sex','ind_trip','ind_secu1',
        'loca_traffic_circul','loca_road_gradient','loca_road_view','loca_accident','loca_road_surface_cond',
        'veh_cat','veh_fixed_obstacle','veh_moving_obstacle','veh_impact','veh_maneuver','veh_motor'
    ]


    # Check which columns are missing
    all_cols = cols_cat + cols_ordinal + cols_nominal + cols_quant + cols_cyclic
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        print("⚠️ Missing columns (ignored):", missing_cols)

    return {
        "categorical": [c for c in cols_cat if c in df.columns],
        "quantitative": [c for c in cols_quant if c in df.columns],
        "ordinal": [c for c in cols_ordinal if c in df.columns],
        "nominal": [c for c in cols_nominal if c in df.columns], 
        "cyclic_encoder": [c for c in cols_cyclic if c in df.columns],
        "target_encoder": [c for c in cols_target_encoder if c in df.columns],
        "oneHot_encoder": [c for c in cols_one_hot_encoder if c in df.columns]
    }


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def print_col_categories(cols_dict: dict) -> None:
    """
    Print all column categories in a readable form.
    
    Parameters
    ----------
    cols_dict : dict
        Dictionary returned by distinguish_cols(df)
    """
    for category, cols in cols_dict.items():
        print(f"\n📂 {category.capitalize()} columns ({len(cols)}):")
        if cols:
            for col in cols:
                print(f"   • {col}")
        else:
            print("   ⚠️ None found in DataFrame")


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def scale_quant_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale quantitative variables in the DataFrame.
    
    - MinMaxScaler is applied to 'loca_max_speed'
    - MinMaxScaler is applied to 'loca_road_count'
    - RobustScaler is applied to 'loca_road_lanes'
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with scaled quantitative variables
    """
    # Initialize scalers
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    # Apply MinMaxScaler to loca_max_speed if present
    if 'loca_max_speed' in df.columns:
        df['loca_max_speed'] = minmax_scaler.fit_transform(df[['loca_max_speed']])
    else:
        print("⚠️ Column 'loca_max_speed' not found, skipping scaling.")
    
    # Apply MinMaxScaler to loca_road_count if present
    if 'loca_road_count' in df.columns:
        df['loca_road_count'] = minmax_scaler.fit_transform(df[['loca_road_count']])
    else:
        print("⚠️ Column 'loca_road_count' not found, skipping scaling.")
    
    # Apply RobustScaler to loca_road_lanes if present
    if 'loca_road_lanes' in df.columns:
        df['loca_road_lanes'] = robust_scaler.fit_transform(df[['loca_road_lanes']])
    else:
        print("⚠️ Column 'loca_road_lanes' not found, skipping scaling.")
    
    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class QuantScaler(BaseEstimator, TransformerMixin):
    """
    Scale quantitative variables:
    - MinMaxScaler for 'loca_max_speed'
    - MinMaxScaler for 'loca_road_count'
    - RobustScaler for 'loca_road_lanes'
    """

    def __init__(self):
        self.minmax_speed = MinMaxScaler()
        self.minmax_count = MinMaxScaler()
        self.robust_lanes = RobustScaler()
        self.fitted_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Fit scalers only on training data
        if 'loca_max_speed' in X.columns:
            self.minmax_speed.fit(X[['loca_max_speed']])
            self.fitted_cols_.append('loca_max_speed')
        if 'loca_road_count' in X.columns:
            self.minmax_count.fit(X[['loca_road_count']])
            self.fitted_cols_.append('loca_road_count')
        if 'loca_road_lanes' in X.columns:
            self.robust_lanes.fit(X[['loca_road_lanes']])
            self.fitted_cols_.append('loca_road_lanes')
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if 'loca_max_speed' in self.fitted_cols_:
            X['loca_max_speed'] = self.minmax_speed.transform(X[['loca_max_speed']])
        if 'loca_road_count' in self.fitted_cols_:
            X['loca_road_count'] = self.minmax_count.transform(X[['loca_road_count']])
        if 'loca_road_lanes' in self.fitted_cols_:
            X['loca_road_lanes'] = self.robust_lanes.transform(X[['loca_road_lanes']])
        return X


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def rearrange_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroup classes of categorical variables with high cardinality
    for veh_cat, veh_fixed_obstacle, veh_moving_obstacle, and veh_maneuver.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with regrouped categorical variables
    """
    # --- veh_cat ---
    if 'veh_cat' in df.columns:
        df['veh_cat'] = df['veh_cat'].replace(10, 7)   # Light vehicle regroup
        df['veh_cat'] = df['veh_cat'].replace(80, 1)   # Bicycle/E-Bike
        df['veh_cat'] = df['veh_cat'].replace([2, 31], 33)   # Motorcycle/Moped
        df['veh_cat'] = df['veh_cat'].replace([13, 15], 14)  # Heavy goods vehicle
        df['veh_cat'] = df['veh_cat'].replace([38, 39, 40], 37)  # Public Transport
        df['veh_cat'] = df['veh_cat'].replace([16, 20, 21], 17)  # Tractor
        df['veh_cat'] = df['veh_cat'].replace(3, 99)   # Other Vehicle
        df['veh_cat'] = df['veh_cat'].replace(60, 50)  # EDP
        df['veh_cat'] = df['veh_cat'].replace([32, 34, 35, 36, 41, 42, 43], 30)  # Scooter/3WD/Quad
    else:
        print("⚠️ Column 'veh_cat' not found, skipping regrouping.")

    # --- veh_fixed_obstacle ---
    if 'veh_fixed_obstacle' in df.columns:
        df['veh_fixed_obstacle'] = df['veh_fixed_obstacle'].replace(3, 4)  # Concrete/Metal Barrier
        df['veh_fixed_obstacle'] = df['veh_fixed_obstacle'].replace([7, 9, 10, 11, 12, 14, 15, 16], 5)  # Other barrier
    else:
        print("⚠️ Column 'veh_fixed_obstacle' not found, skipping regrouping.")

    # --- veh_moving_obstacle ---
    if 'veh_moving_obstacle' in df.columns:
        df['veh_moving_obstacle'] = df['veh_moving_obstacle'].replace([4, 5, 6], 9)  # Other
    else:
        print("⚠️ Column 'veh_moving_obstacle' not found, skipping regrouping.")

    # --- veh_maneuver ---
    if 'veh_maneuver' in df.columns:
        df['veh_maneuver'] = df['veh_maneuver'].replace(12, 11)  # Changing lanes left/right
        df['veh_maneuver'] = df['veh_maneuver'].replace(14, 13)  # Offset left/right
        df['veh_maneuver'] = df['veh_maneuver'].replace(16, 15)  # Turning left/right
        df['veh_maneuver'] = df['veh_maneuver'].replace(18, 17)  # Overtaking left/right
        df['veh_maneuver'] = df['veh_maneuver'].replace([4, 10, 20, 22, 24], 98)  # Others less severe
        df['veh_maneuver'] = df['veh_maneuver'].replace([3, 6, 7, 8, 21, 25], 99)  # Others highly severe
    else:
        print("⚠️ Column 'veh_maneuver' not found, skipping regrouping.")

    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def encode_one_hot(df, cols):
    """
    One-hot encode specified categorical columns directly in the input DataFrame.
    Missing columns are ignored. Uses pd.concat to avoid fragmentation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (will be modified directly).
    cols : list
        List of categorical columns to one-hot encode.
    
    Returns
    -------
    pd.DataFrame
        The same DataFrame with one-hot encoded columns replacing the originals.
    """
    
    # Filter out missing columns
    existing_cols = [c for c in cols if c in df.columns]
    
    if not existing_cols:
        print("No valid columns found for one-hot encoding.")
        return df
    
    # Check shape before encoding
    print("Original shape:", df.shape)
    
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform only existing columns
    encoded_array = encoder.fit_transform(df[existing_cols])
    
    # Convert back to DataFrame with proper column names
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(existing_cols),
        index=df.index
    )
    
    # Drop original categorical columns and concatenate encoded ones in one go
    df.drop(columns=existing_cols, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    # Check shape after encoding
    print("Encoded shape:", df.shape)
    print("Columns encoded:", existing_cols)
    
    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def encode_trigonomeric(df):
    """
    Encode acc_month (1–12) and acc_hour (0–23) into sine/cosine pairs.
    Modifies the DataFrame directly.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'acc_month' and 'acc_hour'.
    
    Returns
    -------
    pd.DataFrame
        The same DataFrame with trigonometric encodings replacing the originals.
    """
    
    # Check shape before encoding
    print("Original shape:", df.shape)
    
    # Encode acc_month (1–12)
    if 'acc_month' in df.columns:
        df['acc_month_sin'] = np.sin(2 * np.pi * df['acc_month'] / 12)
        df['acc_month_cos'] = np.cos(2 * np.pi * df['acc_month'] / 12)
        df.drop(columns=['acc_month'], inplace=True)
    else:
        print("Column 'acc_month' not found, skipped.")
    
    # Encode acc_hour (0–23)
    if 'acc_hour' in df.columns:
        df['acc_hour_sin'] = np.sin(2 * np.pi * df['acc_hour'] / 24)
        df['acc_hour_cos'] = np.cos(2 * np.pi * df['acc_hour'] / 24)
        df.drop(columns=['acc_hour'], inplace=True)
    else:
        print("Column 'acc_hour' not found, skipped.")
    
    # Check shape after encoding
    print("Encoded shape:", df.shape)
    
    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def encode_target(df, cols):
    """
    Apply target encoding to specified categorical columns using 'ind_severity' as target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns and 'ind_severity'.
    cols : list
        List of categorical columns to encode.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with target-encoded columns replacing the originals.
    """
    
    # Ensure target column exists
    if "ind_severity" not in df.columns:
        raise ValueError("Target column 'ind_severity' not found in DataFrame.")
    
    # Define target
    y = df["ind_severity"]
    
    # Initialize encoder
    encoder = TargetEncoder(cols=cols, handle_unknown="ignore")
    
    # Copy DataFrame to avoid chained assignment issues
    df = df.copy()
    
    # Fit and transform specified columns
    df[cols] = encoder.fit_transform(df[cols], y)
    
    print("Target-encoded columns:", cols)
    print("New shape:", df.shape)
    
    return df
