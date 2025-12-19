"""
The library provides five key functions for handling missing values
in pandas DataFrames, focusing on distribution-preserving imputation 
for categorical and quantitative variables.

Use the methods based on data type and available correlation.

Recommended workflow:
1 check data types
    quantitative -> impute_quant_by_cat
    qualitative  -> see step 2 or 3
2 supervised
    prioritize fit & apply rules
    impute_missing_cat_by_target
3 unsupervised / simple
    replace_missing_cat_keep_prop

"""
import time
import logging
from typing import List

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def impute_quant_by_cat(df, quant_var, cat_var, random_state=42):
    """
    Quantitative Variable Imputation
    
    Impute missing values in a quantitative variable using the distribution
    conditional on a categorical variable. 
    If conditional distribution is unavailable, fall back to  
    global distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    quant_var : str
        Quantitative variable with missing values.
    cat_var : str
        Categorical variable correlated with quant_var.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with imputed quant_var.
    """
    rng = np.random.default_rng(random_state)
    missing_mask = df[quant_var].isna()
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print(f"✅ No missing values in '{quant_var}'.")
        return df
    
    print(f"🔍 Found {missing_count} missing values in '{quant_var}'. Imputing...")
    
    # Global distribution (non-missing values)
    global_vals = df.loc[~missing_mask, quant_var].values
    
    for idx in df[missing_mask].index:
        cat_value = df.loc[idx, cat_var]
        
        # Conditional distribution for this category
        cond_vals = df.loc[(df[cat_var] == cat_value) & (~df[quant_var].isna()), quant_var].values
        
        if len(cond_vals) > 0:
            df.loc[idx, quant_var] = rng.choice(cond_vals)
        else:
            # Fallback to global distribution
            df.loc[idx, quant_var] = rng.choice(global_vals)
    
    print(f"✅ Finished imputing '{quant_var}'.")
    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def replace_missing_cat_keep_prop(df: pd.DataFrame, var_list: list, random_state: int = 42) -> pd.DataFrame:
    """
    Simple Categorical Imputation
    
    Replace missing values in categorical variables while keeping
    class proportions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    var_list : list of str
        List of categorical variable names to process.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with imputed categorical variables.
    """
    rng = np.random.default_rng(random_state)
    skipped_missing = []   # columns with no missing values
    skipped_not_found = [] # columns not found in df

    for feat in var_list:
        if feat not in df.columns:
            skipped_not_found.append(feat)
            continue
        
        missing_count = df[feat].isna().sum()
        
        if missing_count == 0:
            skipped_missing.append(feat)
            continue
        
        # Step 1: Get distribution of non-missing values
        probs = df[feat].value_counts(normalize=True, dropna=True)
        
        # Step 2: Sample replacements for NaNs
        replacement = rng.choice(probs.index.to_numpy(), size=missing_count, p=probs.to_numpy())
        
        # Step 3: Fill missing values
        df.loc[df[feat].isna(), feat] = replacement
        
        print(f"🔄 Replaced {missing_count} missing values in '{feat}'.")
    
    # Print skipped columns once at the end
    if skipped_not_found:
        print(f"⚠️ Skipped (not found in DataFrame): {skipped_not_found}")
    if skipped_missing:
        print(f"✅ Skipped (no missing values): {skipped_missing}")
    
    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def impute_missing_cat_by_target(df: pd.DataFrame, var_list: list, target: str, random_state: int = 42) -> pd.DataFrame:
    """
    Target-Conditional Categorical
    
    Replace missing values in categorical variables using the distribution
    conditional on the target variable. If conditional distribution is unavailable,
    fall back to the global distribution of the variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    var_list : list of str
        List of categorical variable names to process.
    target : str
        Target variable name (categorical or numeric class).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with imputed categorical variables.
    """
    start_time = time.time()
    rng = np.random.default_rng(random_state)
    
    skipped_not_found = []  # columns not in df
    skipped_missing = []    # columns with no missing values

    for feat in var_list:
        if feat not in df.columns:
            skipped_not_found.append(feat)
            continue
        
        missing_mask = df[feat].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            skipped_missing.append(feat)
            continue
        
        # Global distribution
        global_probs = df[feat].value_counts(normalize=True, dropna=True)
        
        # Loop through missing rows
        for idx in df[missing_mask].index:
            target_value = df.loc[idx, target]
            
            # Conditional distribution given target_value
            cond_probs = df.loc[df[target] == target_value, feat].value_counts(normalize=True, dropna=True)
            
            if not cond_probs.empty:
                # Sample from conditional distribution
                df.loc[idx, feat] = rng.choice(cond_probs.index.to_numpy(), p=cond_probs.to_numpy())
            else:
                # Fallback to global distribution
                df.loc[idx, feat] = rng.choice(global_probs.index.to_numpy(), p=global_probs.to_numpy())
        
        print(f"🔄 Replaced {missing_count} missing values in '{feat}'.")
    
    end_time = time.time()
    
    # Print skipped columns once at the end
    if skipped_not_found:
        print(f"⚠️ Skipped (not found in DataFrame): {skipped_not_found}")
    if skipped_missing:
        print(f"✅ Skipped (no missing values): {skipped_missing}")
    
    print(f"⏱️ Total computational time: {end_time - start_time:.4f} seconds")
    
    return df


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def fit_impute_rules(X_train: pd.DataFrame, y_train: pd.Series, var_list: list, random_state: int = 42):
    """
    Supervised Rule-Based System
    
    Learn conditional distributions for categorical variables based on target.
    Returns a dict of {feature: {class: distribution}} plus global distributions.
    """
    rng = np.random.default_rng(random_state)
    rules = {}
    
    for feat in var_list:
        if feat not in X_train.columns:
            continue
        
        rules[feat] = {
            "global": X_train[feat].value_counts(normalize=True, dropna=True).to_dict(),
            "conditional": {}
        }
        
        # Conditional distributions per target class
        for cls in y_train.unique():
            cond_probs = X_train.loc[y_train == cls, feat].value_counts(normalize=True, dropna=True)
            if not cond_probs.empty:
                rules[feat]["conditional"][cls] = cond_probs.to_dict()
    
    return rules


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------



def apply_impute_rules(X: pd.DataFrame, y: pd.Series | None, rules: dict, random_state: int = 42):
    """
    Apply learned imputation rules to a dataset.
    - If y is provided (train set), use conditional distributions.
    - If y is None (test set), use global distributions only.
    """
    rng = np.random.default_rng(random_state)
    X = X.copy()
    
    for feat, feat_rules in rules.items():
        if feat not in X.columns:
            continue
        
        missing_mask = X[feat].isna()
        if not missing_mask.any():
            continue
        
        for idx in X[missing_mask].index:
            if y is not None:  # training set
                cls = y.loc[idx]
                cond_probs = feat_rules["conditional"].get(cls, {})
                if cond_probs:
                    choices, probs = zip(*cond_probs.items())
                    X.loc[idx, feat] = rng.choice(choices, p=probs)
                    continue
            # fallback to global distribution
            global_probs = feat_rules["global"]
            choices, probs = zip(*global_probs.items())
            X.loc[idx, feat] = rng.choice(choices, p=probs)
    
    return X


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
    
    # 1. scaling: KNN is sensitive to outliers & scaling
    scaler = StandardScaler()
    
    df_scaled = pd.DataFrame(
        scaler.fit_transform( df_numeric ), 
        columns=df_numeric.columns,
        index=df_numeric.index
    )

    # 2. imputation:
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

    # 3. inverse scaling: transform values back to original scale
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
    X: pd.DataFrame, 
    columns: List[str], 
    new_category_label
):
    """
    Replaces missing values (NaNs) in one or more categorical columns
    with a constant label representing an 'unkown' category.
    
    Example:
    categorical variable 'ambient lightning' has modalities:
    { 0: :unknown, 1: :daylight, 2: :twilight, 3: :night }
    
    Replaces all missing values with 0, representing 'unknown'

    Args:
        X (pd.DataFrame): Input DataFrame
        columns (list[cols]): List of categorical columns to impute.
        new_category_label (int or str, optional): Label used for 'unknown'. Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame with imputed categorical columns.
    """
    
    # Identify the unique NaN representation (np.nan) or other missing strings 
    # and replace them with the constant label
    
    # uses fillna() directly on specific column
    # applying the constant label
    X[columns] = X[columns].fillna( new_category_label )
    
    return X
