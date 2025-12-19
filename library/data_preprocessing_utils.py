""" 
author: Michael Munz

Improved version of cleaning_utils.
A collection of robust, reusable data preprocessing utilities for 
machine learning workflows.

Includes functions for scaling, encoding (one-hot, target, cyclical), 
regrouping categorical variables, and column categorization to 
streamline data preparation and feature engineering.
"""
# ------
# IMPORT
# ------
import warnings
import logging
import sys
sys.path.append( '../../settings_config' )
from data_preprocessing_config import (
    IRRELEVANT_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_IMPUTE_VALUE,
    CATEGORICAL_MODE_IMPUTE_COLUMNS,
    ORDINAL_COLUMNS,
    NOMINAL_COLUMNS,
    QUANTITATIVE_COLUMNS,
    QUANTITATIVE_SCALING_COLUMNS,
    QUANTITATIVE_TO_QUALITATIVE_ORDINAL,
    CYCLIC_COLUMNS,
    TARGET_ENCODER_COLUMNS,
    TARGET_IMPACT_ENCODER_COLUMNS,
    ONE_HOT_ENCODER_COLUMNS,
    REGROUP_RULES,
    QUALITATIVE_REDUCING_MODALITIES_DICT,
)

from copy import deepcopy
from typing import (
    Dict,
    List,
    Optional,
    Tuple
)
from collections import defaultdict
from rich.console import Console
from rich.table import Table

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)

from category_encoders import TargetEncoder


# ---------------
# GENERIC
# ---------------
def categorize_dataframe_columns(
    df: pd.DataFrame, 
    column_sets: Optional[Dict[str, List[str]]]=None
) -> Dict[str, List[str]]:
    """
    Categorize DataFrame columns for preprocessing into groups of:
    - qualitative (categorical)
    - ordinal
    - nominal
    - quantitative (numerical)
    - cyclic
    - target_encoder
    - oneHot_encoder
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column_sets : dict[str, list[str]], optional
        Custom column definitions; defaults to predefined constants
    
    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys: 'categorical', 'ordinal', 'nominal', 'quantitative',
        'cyclic_encoder', 'target_encoder', 'oneHot_encoder'.
        Only columns present in df are included. Missing columns trigger warnings.
    """
    if df.empty:
        logging.info( "Empty datafame provided; returning empty categories" )
        
        return {
            "categorical": [], 
            "ordinal": [], 
            "nominal": [], 
            "quantitative": [],
            "cyclic_encoder": [], 
            "target_encoder": [], 
            "one_hot_encoder": []
        }
    
    # use defaults or override
    defaults = {
        "categorical": CATEGORICAL_COLUMNS,
        "ordinal": ORDINAL_COLUMNS,
        "nominal": NOMINAL_COLUMNS,
        "quantitative": QUANTITATIVE_COLUMNS,
        "cyclic_encoder": CYCLIC_COLUMNS,
        "target_encoder": TARGET_ENCODER_COLUMNS,
        "target_impact_encoder": TARGET_IMPACT_ENCODER_COLUMNS,
        "one_hot_encoder": ONE_HOT_ENCODER_COLUMNS
    }
    
    col_sets = defaults if column_sets is None else { **defaults, **column_sets }

    df_cols = set( df.columns )
    missing_cols = []    
    result = {}
    
    for category, cols in col_sets.items():
        existing = list( set(cols).intersection(df_cols) )
        result[category] = existing
        missing = set( cols ) - df_cols
        
        if missing:
            missing_cols.extend( missing )
    
    if missing_cols:
        missing_str_cols = [col for col in missing_cols if isinstance(col, str)]
        
        logging.info( "Missing columns ignored: %s",
                      sorted(set(missing_str_cols)) )

    return result


def display_column_categories(
    cols_dict: Dict[str, list[str]], 
    quiet: bool = False,
    console: Console = None
) -> Table | None:
    """
    Display column categories in a professional table format using Rich.
    Elevating data exploration in ML pipelines.
    
    Example
    -------
    1: std terminal output => print_col_categories( cols_dict )
    2: jupyter notebook    => display( print_col_categories(cols_dict, 
                                                            quiet=True) )
    3: custom styling      => console = Console( width=120, 
                                                 style="bold" )
                              print_col_categories( cols_dict, 
                                                    console=console )
    
    Parameters
    ----------
    cols_dict : dict[str, list[str]]
        Dictionary from distinguish_cols() with category -> column lists
    quiet : bool, default False
        If True, don't print; just return the table object
    console : rich.Console, optional
        Custom console for advanced control
    
    Returns
    -------
    rich.Table or None
        Table object for Jupyter or programmatic use
    """
    if not console:
        # init :Console
        console = Console()
    
    #  init :Table
    table = Table( title="Column Categories Summary",
                   show_header=True,
                   header_style="bold magenta" )
    
    table.add_column( "Category",
                      style="cyan",
                      no_wrap=True )
    
    table.add_column( "Columns",
                      style="green" )
    
    table.add_column( "Count",
                      justify="right",
                      style="yellow" )
    
    table.add_column( "% of Total",
                      justify="right",
                      style="bright_blue" )
    
    total_cols = sum(len(cols) for cols in cols_dict.values())
    
    # alphabetical order
    for category, cols in sorted( cols_dict.items() ):  
        col_count = len(cols)
        pct_total = f"{(col_count/total_cols*100):.1f}%" if total_cols > 0 else "0.0%"
        
        col_list = ", ".join(sorted(cols)) if cols else "None found"
        table.add_row(
            category.replace("_", " ").title(),
            col_list,
            str(col_count),
            pct_total
        )
    
    return table


# ---------------
# CLEANING
# ---------------
def drop_irrelevant_columns(
    df: pd.DataFrame, 
    columns: Optional[List[str]]=None,
    inplace: bool=False
) -> pd.DataFrame:
    """
    Remove predefined irrelevant columns from the DataFrame.
    If a column does not exist, ignore the error and print its name.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with accident records.
    columns : list[str], optional
        Columns to drop. If None, uses IRRELEVANT_COLUMNS.
    inplace : bool, default False
        If True, modify the original DataFrame and return it.
        If False, return a new DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns dropped.
    """
    if columns is None:
        columns = IRRELEVANT_COLUMNS

    # check for columns that are missing in DF
    missing_cols = [ col for col in columns if col not in df.columns ]
    
    if missing_cols:
        logging.info( "The following columns were not found & skipped: %s",
                      missing_cols )

    # drop ONLY cols that exist in DF
    existing_cols_to_drop = [ col for col in columns if col in df.columns ]

    if inplace:
        df.drop(
            columns=existing_cols_to_drop,
            inplace=True
        )
        
        return df
    else:
        return df.drop(
            columns=existing_cols_to_drop,
            axis=1
        )


# ---------------
# PRE-PROCESSING
# ---------------
def bin_quantitative_to_ordinal(
    df: pd.DataFrame,
    columns: List[str],
    bins: List[float] | int,
    labels: Optional[List[int]]=None,
    suffix: Optional[str]=None,
    inplace: bool=False
) -> pd.DataFrame:
    """
    Convert quantitative columns to ordinal binned categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
    columns : List[str]
        Quantitative columns to bin
    bins : List[float] | int
        Bin edges or number of bins
    labels : List[int], optional
        Ordinal labels for bins (must match len(bins)-1)
    suffix : str, optional
        If provided, creates new columns as `{col}{suffix}`. 
        Otherwise replaces original columns.
    inplace : bool, default False
        Modify df in-place
    
    Usage:
    ------
    # replace original columns
    df = bin_quantitative_to_ordinal(
        df,
        ['speed', 'distance'],
        bins=5
    )

    # create new binned columns (keep originals)
    df = bin_quantitative_to_ordinal(
        df, 
        ['speed'],
        bins=[0, 30, 60, 100],
        suffix='_binned',
        labels=[0, 1, 2]
    )

    # create new binned columns + custom labels
    df = bin_quantitative_to_ordinal(
        df,
        ['temp'],
        bins=3,
        labels=['cold', 'warm', 'hot'],
        suffix='_cat'
    )
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with binned columns
    """
    df = df.copy() if not inplace else df
    
    for col in columns:
        if col not in df.columns:
            logging.warning( f"Column '{col}' not found in DataFrame, skipping" )
        
            continue
        
        # create new column name
        if suffix:
            new_col = f"{col}{suffix}"
        else:
            new_col = col
            
        # create bins and labels
        if isinstance(bins, int):
            bin_edges = pd.cut( df[col],
                                bins=bins,
                                retbins=True )[1]
            bin_labels = range( bins ) if labels is None else labels
        else:
            bin_edges = bins
            bin_labels = range( len(bins)-1 ) if labels is None else labels
            
        # bin the data
        df[new_col] = pd.cut(
            df[col],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=False
        )
        
        # replace original column if no suffix (drop NaN-aware)
        if not suffix:
            df = df.drop( columns=[col] )
    
    return df


def regroup_categorical_columns(
    df: pd.DataFrame, 
    inplace: bool=False
) -> pd.DataFrame:
    """
    Regroup high-cardinality categorical variables using predefined mapping rules.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    inplace : bool, default False
        If True, modify original DataFrame; otherwise return copy
    
    Returns
    -------
    pd.DataFrame
        DataFrame with regrouped categorical variables
    """
    if df.empty:
        logging.info( "Empty dataframe; no re-grouping applied" )
        
        return df if inplace else df.copy()
    
    df_work = df if inplace else deepcopy(df)
    missing_cols = []
    
    for col, mapping in REGROUP_RULES.items():
        if col in df_work.columns:
            # Convert mapping to flat replace dict for pandas
            replace_dict = {}
            for old_vals, new_val in mapping.items():
                if isinstance(old_vals, (list, tuple)):
                    for old_val in old_vals:
                        replace_dict[old_val] = new_val
                else:
                    replace_dict[old_vals] = new_val
            
            # Count changes for validation
            before_count = df_work[col].value_counts().sum()
            df_work[col] = df_work[col].replace(replace_dict)
            after_count = df_work[col].value_counts().sum()
            
            if before_count != after_count:
                warnings.warn( f"⚠️ Column '{col}': Data loss detected during re-grouping.",
                               UserWarning )
                
        else:
            missing_cols.append(col)
    
    if missing_cols:
        logging.info( "Columns not found, skipping %s",
                      missing_cols )
    
    return df_work


def reducing_modalities(
    df: pd.DataFrame,
    mapping_dicts: Dict,
    suffix: str | None=None,
    drop_original=False
):
    """

    Args:
        df (pd.DataFrame): [description]
        mapping_dicts (Dict): [description]
        suffix (str, optional): [description]. Defaults to None.
        drop_original (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    for col, mapping in mapping_dicts.items():
        if col not in df.columns:
            continue

        if suffix is None:
            # overwrite original column
            df[col] = df[ col ].map( mapping )
        else:
            new_col = f"{ col }{ suffix }"
            df[new_col] = df[ col ].map( mapping )
            
            if drop_original:
                df = df.drop( columns=[col] )
    
    return df


# ---------------------------------
# SCALING (STANDARDIZE / NORMALIZE)
# ---------------------------------
def _apply_scaler_to_columns(
    df: pd.DataFrame, 
    columns: List[str], 
    scaler, 
    inplace: bool
) -> pd.DataFrame:
    """Private helper to apply any sklearn scaler to columns."""
    if not inplace:
        df = deepcopy(df)
    
    missing_cols = [ col for col in columns if col not in df.columns ]
    if missing_cols:
        logging.info( "Columns not found, skipping: %s",
                      missing_cols )
    
    valid_cols = [ col for col in columns if col in df.columns ]
    if not valid_cols:
        return df
    
    for col in valid_cols:
        df[col] = scaler.fit_transform( df[[col]] )
    
    return df


def apply_standard_scaler(
    df: pd.DataFrame, 
    columns: List[str], 
    inplace: bool=False
) -> pd.DataFrame:
    """
    Apply StandardScaler to specified columns.
    
    Example
    -------
    df = apply_standard_scaler(df, ['feature1', 'feature2'])
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list[str]
        Columns to scale with StandardScaler
    inplace : bool, default False
        If True, modify original DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with StandardScaler applied
    """
    scaler = StandardScaler()
    
    return _apply_scaler_to_columns( df, 
                                     columns, 
                                     scaler, 
                                     inplace )


def apply_minmax_scaler(
    df: pd.DataFrame, 
    columns: List[str], 
    inplace: bool=False
) -> pd.DataFrame:
    """
    Apply MinMaxScaler to specified columns.
    
    Example
    -------
    df = apply_minmax_scaler(df, ['loca_max_speed', 'loca_road_count'])
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list[str]
        Columns to scale with MinMaxScaler
    inplace : bool, default False
        If True, modify original DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with MinMaxScaler applied
    """
    scaler = MinMaxScaler()
    
    return _apply_scaler_to_columns( df, 
                                     columns, 
                                     scaler, 
                                     inplace )


def apply_robust_scaler(
    df: pd.DataFrame, 
    columns: List[str], 
    inplace: bool=False
) -> pd.DataFrame:
    """
    Apply RobustScaler to specified columns (outlier-resistant).
    
    Scales features using median & IQR (0.25, 0.75 percentiles)
    instead of mean/std; making it resistant to outliers.
    
    UC
    --
    - outlier-heavy data
    - skewed distribution
    
    Example
    -------
    df2 = apply_robust_scaler(df, ['loca_road_lanes'])
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list[str]
        Columns to scale with RobustScaler
    inplace : bool, default False
        If True, modify original DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with RobustScaler applied
    """
    scaler = RobustScaler()
    
    return _apply_scaler_to_columns( df, 
                                     columns, 
                                     scaler, 
                                     inplace )


# ---------------
# IMPUTING
# ---------------
def impute_numerical_by_knn(
    X: pd.DataFrame,
    columns: List[str],
    n_neighbors: int=5
) -> pd.DataFrame:
    """
    Imputes missing values in specified numerical columns using KNNImputer.
    Data is standardized before imputation, as KNN is sensitive to scaling.
    
    Args:
        X (pd.DataFrame): [description]
        columns (list): [description]
        n_neighbors (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    df = X.copy()
    
    # select only relevant numerical subset for imputation
    numeric_cols = [ c for c in columns
                     if c in df.columns and
                     pd.api.types.is_numeric_dtype(df[c]) ]

    if not numeric_cols:
        # nothing to impute
        return df

    df_numeric = df[ numeric_cols ].copy()
    
    # 1. scaling
    # KNN is sensitive to outliers & scaling
    scaler = StandardScaler()
    
    df_scaled = pd.DataFrame(
        scaler.fit_transform( df_numeric ), 
        columns=df_numeric.columns,
        index=df_numeric.index
    )

    # 2. imputation
    # init :KNNImputer -> use AVG of :kNN to fill NaNs
    imputer = KNNImputer( n_neighbors=n_neighbors )
    
    # apply fit & transform
    imputed_array = imputer.fit_transform( df_scaled )
    
    # convert back to DataFrame, keeping column names & index
    df_imputed_scaled = pd.DataFrame(
        imputed_array,
        columns=df_numeric.columns,
        index=df_numeric.index
    )

    # 3. inverse scaling
    # transform values back to original scale
    df_imputed_original_scale = pd.DataFrame(
        scaler.inverse_transform( df_imputed_scaled ),
        columns=df_numeric.columns,
        index=df_numeric.index
    )

    # replace original columns in original DataFrame with imputed values
    X[columns] = df_imputed_original_scale
    
    return X


def impute_numerical_by_regression(
    X: pd.DataFrame, 
    target_col: str, 
    predictor_cols: List[str]
) -> pd.DataFrame:
    """
    Uses regression analysis imputation.
    Advanced technique, treats variable with missing values as a
    dependent target variable :y and other complete columns as independent
    explanatory variables :X.
    Linear regression model is used to predict missing entries in :y.
    
    Imputes missing values in target_col using Linear Regression, 
    based on values in predictor_cols.

    Args:
        X (pd.DateFrame): [description]
        target_col (pd.Series): a numerical variable with missing values (NaNs)
        predictor_cols (pd.Series): features associated with 'target_col' and
                                    only complete rows

    Raises:
        ValueError: [description]

    Returns:
        [pd.DataFrame]: returns DF with imputed values in 'target_col'
    """
    df_copy = X.copy()

    # 1. split data into observed (target is known) and 
    # missing (target is NaN)
    observed_data = df_copy.dropna( subset=[target_col] )
    missing_data = df_copy[ df_copy[target_col].isna() ]
    
    # check if there are missing values to impute
    if missing_data.empty:
        logging.info( "No missing values found in %s",
                      target_col )
        
        return X
        
    # check if predictor columns are complete in observed data
    # (necessary for training)
    if observed_data[predictor_cols].isnull().any().any():
        # any predictor with NaNs MUST be imputed beforehand or excluded
        # in real pipeline, predictors should be imputed first
        raise ValueError( "Predictor columns must be complete for training imputer model." )

    # 2. define training set (from observed samples)
    X_train = observed_data[ predictor_cols ]
    y_train = observed_data[ target_col ]

    # 3. define prediction set (where to fill the NaN)
    X_predict = missing_data[ predictor_cols ]

    # 4. train linear regression model
    # linear regression finds coefficients (m & b) for linear relationship
    model = LinearRegression()
    
    model.fit( X_train,
               y_train )

    # 5. predict missing values
    y_predicted = model.predict( X_predict )

    # 6. fill missing values in original DataFrame
    df_copy.loc[missing_data.index, target_col] = y_predicted

    return df_copy


def impute_categorical_by_category(
    df: pd.DataFrame,
    columns: List[str]=None,
    fill_value: int=0
) -> pd.DataFrame:
    """
    For each column in columns list missing values are filled by 'unknown' label
    
    Args:
        df (pd.DataFrame): [description]
        columns (List[str], optional): [description]. Defaults to None.
        fill_value (int, optional): [description]. Defaults to '0'.

    Returns:
        pd.DataFrame: [description]
    """
    for col in columns:
        df[col] = df[col].fillna( fill_value )
        
    return df


def impute_categorical_by_mode(
    df: pd.DataFrame,
    columns: List[str]=None,
) -> pd.DataFrame:
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        mode_val = df[col].mode(dropna=True)
        
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])
    
    return df


def group_columns_by_impute_value(mapping: Dict[str, int]):
    """
    Groups the 'mapping' dictionary by impute value
    
    Usage:
        groups = group_columns_by_impute_value(CATEGORICAL_IMPUTE_VALUE)
        groups[0]

    Args:
        mapping (Dict[str, int]): { 'col1': 0, 'col2': 4, 'col3': 5 }

    Returns:
        [type]: [description]
    """
    groups = defaultdict(list)
    
    for col, val in mapping.items():
        groups[val].append( col )
    
    return groups


# ---------------
# ENCODING
# ---------------
def apply_ordinal_encoding(
    df: pd.DataFrame,
    cols: List[str],
    inplace: bool=False
) -> pd.DataFrame:
    """
    Apply ordinal encoding to specified categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    cols : list of str
        Columns to ordinal encode.
    inplace : bool, default False
        If True, modify original DataFrame; otherwise return a copy.
    
    Usage
    -----
    df_encoded = apply_ordinal_encoding(
        df,
        cols=['col1', 'col2'],
        inplace=False
    )

    Returns
    -------
    pd.DataFrame
        DataFrame with ordinal-encoded columns replacing originals.
    """
    if not inplace:
        df = df.copy()

    existing_cols = [ c for c in cols if c in df.columns ]
    missing_cols = set( cols ) - set( existing_cols )

    if missing_cols:
        logging.info( "Columns not found & skipped: %s",
                      missing_cols )

    if not existing_cols:
        logging.info( "No valid columns found for ordinal encoding" )
        
        return df

    # init :OrdinalEncoder
    encoder = OrdinalEncoder()

    df[existing_cols] = encoder.fit_transform( df[existing_cols] )

    return df


def apply_one_hot_encoding(
    df: pd.DataFrame, 
    cols: List[str], 
    inplace: bool=False,
    drop_first: bool=False,
    handle_unknown: str='ignore'
) -> pd.DataFrame:
    """
    One-hot encode specified categorical columns using scikit-learn OneHotEncoder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    cols : list[str]
        Categorical columns to one-hot encode
    inplace : bool, default False
        If True, modify original DataFrame
    drop_first : bool, default False
        Drop first category to avoid multicollinearity
    handle_unknown : str, default 'ignore'
        OneHotEncoder handle_unknown parameter
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded columns
    """
    if df.empty:
        logging.info( "Empty dataframe; no encoding applied" )
        
        return df if inplace else df.copy()
    
    df_work = df if inplace else deepcopy(df)
    existing_cols = [c for c in cols if c in df_work.columns]
    
    if not existing_cols:
        logging.info( "No valid columns found for OHE" )
        
        return df_work
    
    # encode each column separately
    encoded_dfs = []
    encoder = OneHotEncoder(
        drop=drop_first, 
        sparse_output=False, 
        handle_unknown=handle_unknown
    )
    
    for col in existing_cols:
        # handle NaN consistently
        col_data = df_work[[col]].fillna( 'missing' )
        encoded = encoder.fit_transform( col_data )
        encoded_cols = encoder.get_feature_names_out( [col] )
        encoded_df = pd.DataFrame( encoded, 
                                   columns=encoded_cols, 
                                   index=df_work.index )
        encoded_dfs.append( encoded_df )
    
    # Safe concatenation and cleanup
    encoded_combined = pd.concat( encoded_dfs, 
                                  axis=1 )
    df_work = df_work.drop( columns=existing_cols )
    df_work = pd.concat( [df_work, encoded_combined], 
                         axis=1 )
    
    n_original = len( existing_cols )
    n_encoded = encoded_combined.shape[1]
    
    logging.info( "OHE: %s cols -> %s cols (+%s)",
                  n_original,
                  n_encoded,
                  n_encoded - n_original)
    
    return df_work


def apply_target_encoding(
    df: pd.DataFrame, 
    cols: List[str], 
    target_col: str = "ind_severity",
    inplace: bool = False
) -> pd.DataFrame:
    """
    Apply target encoding to specified categorical columns using a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns and target column.
    cols : list of str
        Columns to target encode.
    target_col : str, default "ind_severity"
        Target column name.
    inplace : bool, default False
        If True, modify original DataFrame; otherwise return a copy.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with target-encoded columns replacing originals.
    
    Notes
    -----
    Requires `category_encoders` package: `pip install category_encoders`.
    """
    if target_col not in df.columns:
        raise ValueError( f"Target column '{target_col}' not found in DataFrame." )
    
    df_work = df if inplace else deepcopy(df)
    
    existing_cols = [c for c in cols if c in df_work.columns]
    missing_cols = set(cols) - set(existing_cols)
    if missing_cols:
        logging.info( "Columns not found & skipped: %s",
                      missing_cols )
    if not existing_cols:
        logging.info( "No valid columns found for target encoding" )
        
        return df_work
    
    y = df_work[target_col]
    encoder = TargetEncoder( cols=existing_cols, 
                             handle_unknown="ignore" )
    
    df_work[existing_cols] = encoder.fit_transform( df_work[existing_cols], 
                                                    y )
    
    logging.info( "Target-encoded columns: %s",
                  existing_cols )
    logging.info( "Dataframe shape after encoding: %s",
                  df_work.shape )
    
    return df_work


def apply_target_impact_encoding(
    df: pd.DataFrame,
    cols: List[str],
    target_col: str,
    smoothing: float=1.0,
    min_samples_leaf: int=1,
    inplace: bool=False
) -> pd.DataFrame:
    """
    Apply target/impact encoding to specified categorical columns with
    optional smoothing.
    
    UC
    --
    for very high-cardinality variables, like 'acc_municipality'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns and target column.
    cols : list of str
        Columns to target encode.
    target_col : str
        Target column name.
    smoothing : float, default 1.0
        Smoothing effect to balance categorical average vs global average; 
        higher means stronger regularization.
    min_samples_leaf : int, default 1
        Minimum samples needed to take category average into account for smoothing.
    inplace : bool, default False
        If True, modify original DataFrame; otherwise return a copy.
    
    Example:
    df_encoded = apply_target_encoding_with_smoothing(
        df,
        cols=['acc_municipality'],
        target_col='ind_severity',
        smoothing=10,
        min_samples_leaf=20,
        inplace=False
    )

    Returns
    -------
    pd.DataFrame
        DataFrame with target-encoded columns replacing originals.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if not inplace:
        df = df.copy()

    global_mean = df[target_col].mean()

    for col in cols:
        if col not in df.columns:
            logging.info( "Column %s not found, skipping target encoding",
                          col )
            
            continue

        # Compute target mean per category & category counts
        agg = df.groupby( col )[target_col].agg( ['mean', 'count'] )
        counts = agg['count']
        means = agg['mean']

        # Compute smoothing
        smoothing_factor = 1 / ( 1 + np.exp(-(counts - min_samples_leaf) / smoothing) )

        # Compute smoothed means: weighted average of global mean and category mean
        smooth_means = global_mean * ( 1 - smoothing_factor ) + means * smoothing_factor

        # Map smoothed means to the column
        df[col] = df[col].map( smooth_means ).fillna( global_mean )

    return df


def encode_cyclical_features(
    df: pd.DataFrame,
    columns: List[Tuple[str, int]],
    inplace: bool=False
) -> pd.DataFrame:
    """
    Encode cyclical columns into sine/cosine pairs, such as
    month (1 – 12) and hour (0 – 23)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[Tuple[str, int]]
        List of tuples with (column_name, period) specifying cyclical columns and their periods
    inplace : bool, default False
        If True, modify the original DataFrame. Otherwise, return a modified copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with trigonometric encodings replacing original columns.

    Usage
    -----
    df_encoded = encode_cyclical_features(
        df,
        columns=[('acc_month', 12), ('acc_hour', 24)],
        inplace=False
    )
    """
    def encode_column_cyclic(data: pd.Series, period: int, prefix: str) -> pd.DataFrame:
        """Helper to encode a single cyclical column as sin/cos."""
        sin_col = np.sin( 2 * np.pi * data / period )
        cos_col = np.cos( 2 * np.pi * data / period )
        return pd.DataFrame( {f"{prefix}_sin": sin_col, f"{prefix}_cos": cos_col} )

    if not inplace:
        df = deepcopy(df)

    for col, period in columns:
        if col in df.columns:
            if df[col].isnull().any():
                warnings.warn( f"Column '{col}' contains NaNs; encoding will propagate NaNs.", 
                               UserWarning )
            
            encoded_df = encode_column_cyclic( df[col], 
                                               period, 
                                               col )

            df = pd.concat( [df.drop(columns=[col]), 
                             encoded_df], 
                            axis=1 )
        else:
            logging.info( "Column %s not found, skipping encoding",
                          col )

    return df
