"""
author: Michael Munz

Generic ML Pipeline using 'data_preprocessing_utils.py'
====================================================
preprocessing pipeline uses methods from 'data_preprocessing_utils.py'.

Pipeline uses following data pre-processing workflow:
1. Remove irrelevant columns
2. Regroup high-cardinality categorical variables
3. Impute missing values (conditional on categories/target)
4. Scale quantitative variables
5. Encode cyclical features trigonometrically
6. One-hot encode nominal categoricals
7. Target encode high-cardinality categoricals

Usage:
    1) standalone usage ( direct fit_transform() )
    drop = DropIrrelevantColumns(irrelevant_columns=IRRELEVANT_COLUMNS)
    X_clean = drop.fit_transform(X_train)
    
    2) via pipeline
    pipeline = Pipeline([
        ('drop_irrelevant', DropIrrelevantColumns(irrelevant_columns=IRRELEVANT_COLUMNS)),
        # ...more steps
    ])
    X_processed = pipeline.fit_transform(X_train, y_train)
    
    3) via ColumnTransformer (advanced)
    preprocessor = ColumnTransformer([
        ('drop_cols', DropIrrelevantColumns(irrelevant_columns=IRRELEVANT_COLUMNS), 'drop'),
        # ... other transformers
    ], remainder='passthrough)
"""
import logging
import pandas as pd
import sys
sys.path.append( '../../library' )
sys.path.append( '../../settings_config' )
from data_preprocessing_config import (
    IRRELEVANT_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORICAL_IMPUTE_VALUE,
    CATEGORICAL_MODE_IMPUTE_COLUMNS,
    EXCLUDED_FROM_REGRESSION,
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

from typing import List, Tuple, Optional, Dict, Union

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# custom library
from data_preprocessing_utils import (
    # generic
    categorize_dataframe_columns,
    display_column_categories,
        
    # cleaning
    drop_irrelevant_columns,
    
    # pre-processing
    regroup_categorical_columns,
    reducing_modalities,
    bin_quantitative_to_ordinal,
    
    # scaling (standardize / normalize)
    apply_standard_scaler,
    apply_minmax_scaler,
    apply_robust_scaler,
    
    # imputing
    impute_categorical_by_category,
    impute_categorical_by_mode,
    impute_numerical_by_knn,
    impute_numerical_by_regression,
    group_columns_by_impute_value,
    
    # encoding
    apply_ordinal_encoding,
    apply_one_hot_encoding,
    apply_target_encoding,
    apply_target_impact_encoding,
    encode_cyclical_features
)


# -------
# WRAPPER
# -------
class DropIrrelevantColumns(BaseEstimator, TransformerMixin):
    """Wrapper for drop_irrelevant_columns()."""
    def __init__(self, columns: List[str], inplace: bool=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """new DF with irrelevant columns removed"""
        return drop_irrelevant_columns( df=X, 
                                        columns=self.columns,
                                        inplace=self.inplace )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class ReduceQualitativeModalities(BaseEstimator, TransformerMixin):
    """
    Apply direct mapping from raw categorical codes to reduced modalities.

    For each column in mapping_dicts, creates/overwrites a column
    '<col>_red' containing the mapped integer code.
    
    To overwrite in place (no suffix) set 'suffix=None' and map directly
    onto same column.
    
    To create '*_red' and drop old:
    QualitativeReduceModalities(mapping_dicts, suffix="_red", drop_original=True)
    """
    def __init__(
        self,
        mapping_dicts: Dict,
        suffix: str | None=None,
        drop_original: bool=False
    ):
        self.mapping_dicts = mapping_dicts
        self.suffix = suffix
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        return reducing_modalities( df=X,
                                    mapping_dicts=self.mapping_dicts,
                                    suffix=self.suffix,
                                    drop_original=self.drop_original )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class BinQuantitativeToOrdinal(BaseEstimator, TransformerMixin):
    """
    Apply direct mapping from raw categorical codes to reduced modalities.

    For each column in mapping_dicts, creates/overwrites a column
    '<col>_red' containing the mapped integer code.
    
    To overwrite in place (no suffix) set 'suffix=None' and map directly
    onto same column.
    
    To create '*_red' and drop old:
    QuantitativeToQualitativeOrdinal(mapping_dicts, suffix="_red", drop_original=True)
    """
    def __init__(
        self,
        columns: List[str],
        bins: List[float] | int,
        labels: Optional[List[int]]=None,
        suffix: Optional[str]=None,
        inplace: bool=False
    ):
        self.columns = columns
        self.bins = bins
        self.labels = labels
        self.suffix = suffix
        self.inplace = inplace
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return bin_quantitative_to_ordinal(
            df=X,
            columns=self.columns,
            bins=self.bins,
            labels=self.labels,
            suffix=self.suffix,
            inplace=self.inplace
        )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class RegroupCategoricalColumns(BaseEstimator, TransformerMixin):
    """Wrapper for regroup_categorical_columns."""
    def __init__(self, inplace: bool=False, verbose=True):
        self.inplace = inplace
        self.verbose = verbose
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply re-grouping"""
        return regroup_categorical_columns( df=X, 
                                            inplace=self.inplace )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class EncodeMunicipalityAsId(BaseEstimator, TransformerMixin):
    """
    Encode 'acc_municipality' as a single integer ID using pandas.factorize.
    - Treats each distinct municipality code as a categorical label.
    - Works with mixed formats like '2B010', '75056', etc.
    - Produces one numeric column: 'acc_municipality_id'.
    - Drops the original 'acc_municipality' column.
    """
    def __init__(self, column: str='acc_municipality'):
        self.column = column
        # to store mapping for consistency
        self.classes_ = None
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        
        if self.column in X.columns:
            # factorize once to fix the category → id mapping
            codes, uniques = pd.factorize( X[self.column].astype('string') )
            self.classes_ = uniques.to_list()
        else:
            self.classes_ = []
        
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        if self.column not in X.columns:
            # nothing to do
            return X
        
        # map labels to IDs using the fitted categories
        # pandas.Categorical ensures consistent mapping at transform time
        cat = pd.Categorical( X[self.column].astype('string'),
                             categories=self.classes_ )
        
        # -1 for unseen/NaN
        codes = cat.codes
        
        # choose how to handle unknowns; here keep -1
        X[self.column] = codes.astype( 'int64' )
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class Standardization(BaseEstimator, TransformerMixin):
    """Pipeline wrapper for apply_standard_scaler()."""
    
    def __init__(self, columns: List[str], inplace: bool=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply standard scaling to specified columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError( "Input must be a pandas DataFrame" )
        
        return apply_standard_scaler( df=X,
                                      columns=self.columns,
                                      inplace=self.inplace )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class NominalizationMinMax(BaseEstimator, TransformerMixin):
    """Pipeline wrapper for apply_minmax_scaler()."""
    
    def __init__(self, columns: List[str], inplace: bool=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply min-max nominalization to specified columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError( "Input must be a pandas DataFrame" )
        
        return apply_minmax_scaler( df=X,
                                    columns=self.columns,
                                    inplace=self.inplace )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class NominalizationRobust(BaseEstimator, TransformerMixin):
    """Pipeline wrapper for apply_robust_scaler()."""
    
    def __init__(self, columns: List[str], inplace: bool=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply robust nominalization to specified columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError( "Input must be a pandas DataFrame" )
        
        return apply_robust_scaler( df=X,
                                    columns=self.columns,
                                    inplace=self.inplace )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class OrdinalEncoding(BaseEstimator, TransformerMixin):
    """Wrapper for apply_ordinal_encoding()."""
    def __init__(
        self, 
        columns: List[str], 
        inplace: bool=False
    ):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply ordinal encoding to specified columns"""
        return apply_ordinal_encoding( df=X,
                                       cols=self.columns,
                                       inplace=self.inplace )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class CyclicEncoding(BaseEstimator, TransformerMixin):
    """Wrapper for encode_cyclical_features()."""
    def __init__(self, columns: List[Tuple[str, int]], inplace: bool=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # columns: List[Tuple[str, int]]
        return encode_cyclical_features( df=X,
                                         columns=self.columns,
                                         inplace=self.inplace )

    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class OneHotEncoding(BaseEstimator, TransformerMixin):
    """Wrapper for apply_one_hot_encoding()."""
    def __init__(
        self, 
        columns: List[str], 
        inplace: bool=False, 
        drop_first: bool=False,
        handle_unknown: str='ignore'
    ):
        self.columns = columns
        self.inplace = inplace
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply OHE to specified columns"""
        debugging = False
        
        # debugging
        if debugging:
            for col in self.columns:
                if col in X.columns:
                    print( col,
                           X[col].dtype,
                           X[col].dropna().unique()[:10] )

        return apply_one_hot_encoding( df=X,
                                       cols=self.columns,
                                       inplace=self.inplace,
                                       drop_first=self.drop_first,
                                       handle_unknown=self.handle_unknown )

    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class TargetEncoding(BaseEstimator, TransformerMixin):
    """Wrapper for apply_target_encoding()"""
    def __init__(
        self,
        columns: List[str],
        target_col: str='ind_severity',
        inplace: bool=False
    ):
        self.columns = columns
        self.target_col = target_col
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply target encoding to  specified column"""
        return apply_target_encoding( df=X, 
                                      cols=self.columns,
                                      target_col=self.target_col,
                                      inplace=self.inplace )

    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class TargetImpactEncoding(BaseEstimator, TransformerMixin):
    """Wrapper for apply_target_impact_encoding()"""
    def __init__(
        self,
        columns: List[str],
        target_col: str,
        smoothing: float=1.0,
        min_samples_leaf: int=1,
        inplace: bool=False
    ):
        self.columns = columns
        self.target_col = target_col
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.inplace = inplace
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """
        Apply target encoding to high cardinality column,
        using target/impact encoding
        """
        return apply_target_impact_encoding(
            df=X,
            cols=self.columns,
            target_col=self.target_col,
            smoothing=self.smoothing,
            min_samples_leaf=self.min_samples_leaf,
            inplace=self.inplace
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class ImputeQuantitativeVariableByKNN(BaseEstimator, TransformerMixin):
    """Wrapper for impute_numerical_by_knn()"""
    def __init__(
        self,
        columns: List[str],
        n_neighbors: int=5
    ):
        self.columns = columns
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Impute quantitative target variable by regression using predictor cols"""
        return impute_numerical_by_knn( X=X,
                                        columns=self.columns,
                                        n_neighbors=self.n_neighbors )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class ImputeQuantitativeVariableByRegression(BaseEstimator, TransformerMixin):
    """Impute target column using regression on complete predictor columns."""
    
    def __init__(self, target_col: str, predictor_cols: Optional[List[str]] = None):
        self.target_col = target_col
        # optional manual override
        self.predictor_cols = predictor_cols
    
    def fit(self, X, y=None):
        """Automatically detect complete columns during fit."""
        X = pd.DataFrame(X)

        if self.predictor_cols is None:
            # 1) only complete columns (no NaNs)
            complete_cols = X.columns[X.notna().all()].tolist()

            # 2) remove target
            if self.target_col in complete_cols:
                complete_cols.remove(self.target_col)

            # 3) keep only numeric predictors (dtypes)
            numeric_complete = [
                col for col in complete_cols
                if pd.api.types.is_numeric_dtype(X[col])
            ]

            self.complete_predictors_ = numeric_complete
        else:
            # manual list: keep only complete & numeric
            self.complete_predictors_ = [
                col for col in self.predictor_cols
                if col in X.columns
                and X[col].notna().all()
                and pd.api.types.is_numeric_dtype(X[col])
            ]

        return self
    
    def transform(self, X):
        """Impute using auto-detected complete predictors."""
        X = pd.DataFrame(X)
        
        # use impute_numerical_by_regression with auto-detected complete predictors
        return impute_numerical_by_regression(
            X=X,
            target_col=self.target_col,
            predictor_cols=self.complete_predictors_
        )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class QuantitativeImputer(BaseEstimator, TransformerMixin):
    """
    Run regression imputation first, then KNN only for columns
    that still contain NaNs after regression.
    """
    def __init__(self, regression_imputer, knn_imputer=None, label="quant_impute"):
        self.regression_imputer = regression_imputer
        self.knn_imputer = knn_imputer
        self.label = label

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        
        # Fit all regression imputers
        for name, reg in self.regression_imputer:
            reg.fit(X, y)
            X = reg.transform(X)
        
        # Optionally fit KNN on regression-imputed data
        if self.knn_imputer is not None:
            if X.isna().any().any():
                self.knn_imputer.fit(X, y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        # 1) Regression imputers
        for name, reg in self.regression_imputer:
            X = reg.transform(X)

        # 2) Check NaNs
        total_nans = X.isna().sum().sum()
        
        if total_nans == 0 or self.knn_imputer is None:
            logging.info(
                "[%s] After regression: %d NaNs, KNN %s",
                self.label,
                total_nans,
                "skipped" if self.knn_imputer is None else "not needed"
            )
            return X

        # 3) Run KNN only if NaNs remain
        logging.info(
            "[%s] After regression: %d NaNs remain, applying KNN",
            self.label,
            total_nans
        )
        
        return self.knn_imputer.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class ImputeQualitativeVariablesByCategory(BaseEstimator, TransformerMixin):
    """Wrapper for impute_categorical_by_category()"""
    def __init__(
        self,
        columns: List[Tuple[str, int]],
        fill_value: int
    ):
        self.columns = columns
        self.fill_value = fill_value
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply target encoding to  specified column"""
        return impute_categorical_by_category(
            df=X,
            columns=self.columns,
            fill_value=self.fill_value
        )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class ImputeQualitativeVariablesByMode(BaseEstimator, TransformerMixin):
    """Wrapper for impute_categorical_by_mode()"""
    def __init__(
        self,
        columns: List[Tuple[str, int]]
    ):
        self.columns = columns
    
    def fit(self, X, y=None):
        """Fit is pass-through (stateless wrapper)."""
        return self
    
    def transform(self, X):
        """Apply target encoding to  specified column"""
        return impute_categorical_by_mode(
            df=X,
            columns=self.columns
        )
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray( input_features )


class NaNCheckTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, label: str = ""):
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        
        total_nans = X.isna().sum().sum()
        
        if total_nans > 0:
            logging.warning(
                "[NaN CHECK] %s: %d NaNs, per-column=%s",
                self.label,
                total_nans,
                X.isna().sum()[X.isna().sum() > 0].to_dict(),
            )
        else:
            logging.info("[NaN CHECK] %s: no NaNs", self.label)
        
        return X


# ------------------------------------
# LOW-LEVEL (SPECIFIC) PIPELINE STEPS
# -----------------------------------

# --------
# reducing
# --------
def create_reducing_step(
    mapping_dicts: Dict,
    suffix: str | None=None,
    drop_original: bool=False
) -> Pipeline:
    steps = []
    
    steps.append(
        ( 'reduce_qualitative_modalities',
          ReduceQualitativeModalities(
              mapping_dicts=mapping_dicts,
              suffix=suffix,
              drop_original=drop_original
            )
        )
    )
    
    return Pipeline( steps )


# --------
# binning
# --------
def create_binning_step(
    binning_config: Dict,
    suffix: str | None=None,
    inplace: bool=False
) -> Pipeline:
    """
    Usage:
    -----
    binning_config = {
        'lanes': {
            'columns': ['loca_road_lanes'],
            'bins': [1, 3, 5, float('inf')],
            'labels': [1, 2, 3]
        },
        'speed': {
            'columns': ['loca_max_speed'],
            'bins': [0, 50, 90, float('inf')],
            'labels': [1, 2, 3]
        }
    }
    
    Args:
        binning_config (Dict): { 'cfg_1': { columns: [], bins: [], labels: [] }
                                 'cfg_n': { columns: [], bins: [], labels: [] }  }
        suffix (str, optional): Defaults to None.
        inplace (bool, optional): Defaults to False.

    Returns:
        Pipeline: [description]
    """
    steps = []
    
    for k_name, v_config in binning_config.items():
        steps.append(
            ( f'binning_{k_name}_to_ordinal',
              BinQuantitativeToOrdinal(
                  columns=v_config['columns'],
                  bins=v_config['bins'],
                  labels=v_config['labels'],
                  suffix=suffix,
                  inplace=inplace
              )
            )
        )
    
    return Pipeline( steps )


# --------
# imputing
# --------
def create_quantitative_imputing_step(
    knn: Optional[Dict]=None,
    regression: Optional[Union[List[Dict[str, List[str]]], List[str]]]=None
) -> Pipeline:
    """
    Factory function to create quantitative imputation pipeline.
    
    Args:
        knn: Dict with 'columns': List[str], 'n_neighbors': int (default=5)
             e.g. {'columns': ['loca_max_speed'], 'n_neighbors': 5}
        regression: EITHER:
            1. List of dicts: [{'target_col': str, 'predictor_cols': List[str]}, ...]
            2. List of target column names: ['col1', 'col2'] (auto-detects predictors)
    
    Returns:
        Pipeline: Pipeline with a single QuantitativeImputer step
                  (regression first, then KNN if NaNs remain).
    """
    regression_imputer = []

    # 1) build regression imputers (if any)
    if regression:
        # simple list of target columns → convert to list of dict configs
        if isinstance(regression, list) and all(isinstance(item, str) for item in regression):
            regression = [{"target_col": col} for col in regression]

        for config in regression:
            target_col = config["target_col"]
            
            # None → auto-detect
            # predictor_cols = config.get("predictor_cols", None)
            predictor_cols = [
                c for c in config.get("predictor_cols", None) or []
                if c not in EXCLUDED_FROM_REGRESSION
            ] or None
            
            reg = ImputeQuantitativeVariableByRegression(
                target_col=target_col,
                predictor_cols=predictor_cols,
            )
            
            regression_imputer.append(
                (f"regression_impute_{target_col}", reg)
            )

    # 2) build KNN imputer (if provided)
    knn_imputer = None
    
    if knn:
        columns = knn.get("columns", [])
        n_neighbors = knn.get("n_neighbors", 5)
        
        if columns:
            knn_imputer = ImputeQuantitativeVariableByKNN(
                columns=columns,
                n_neighbors=n_neighbors,
            )

    # 3) wrap both into one meta-transformer
    quant_imputer = QuantitativeImputer(
        regression_imputer=regression_imputer,
        knn_imputer=knn_imputer,
        label="quantitative_imputer",
    )

    # Expose as a standard Pipeline
    return Pipeline( [("quantitative_imputer", quant_imputer)] )


def create_qualitative_imputing_step(
    by_category: Optional[Dict[str, int]]=None,
    by_mode: Optional[List[str]]=None
) -> Pipeline:
    """
    Factory function to create qualitative (categorical) imputing pipeline.

    Args:
        by_category:
            Mapping from column name to category value to use for imputation.
            If None, uses global CATEGORICAL_IMPUTE_VALUE.
        by_mode:
            Imputing columns by mode

    Usage:
        by_category = { col_1: 0, ..., col_n: 5 }
        by_mode = [ 'col_1', ..., 'col_n' ]
    
    Returns:
        Pipeline: Imputation pipeline (only includes active imputation steps)
    """
    steps = []
    
    # 1 impute by category
    if by_category:
        # {fill_value: [col1, col2, ...]}
        groups = group_columns_by_impute_value(by_category)

        for fill_value, cols in groups.items():
            if not cols:
                continue
            
            step_name = f"impute_by_category_{fill_value}"
            
            steps.append(
                ( step_name,
                  ImputeQualitativeVariablesByCategory(
                        columns=cols,
                        fill_value=fill_value,
                    ) )
            )
    
    # 2 impute by mode
    if by_mode:
        steps.append(
            ( 'impute_by_mode',
              ImputeQualitativeVariablesByMode(
                  columns=by_mode
              ) )
        )

    return Pipeline(steps)


def create_imputing_pipeline(
    quantitative: Optional[Dict]=None,
    qualitative: Optional[Dict[str, int]]=None
) -> Pipeline:
    """
    Factory function to create COMPLETE imputation pipeline (quantitative + qualitative).
    
    Args:
        quantitative: Dict with 'knn' and/or 'regression'
            e.g. {
                'knn': {'columns': [...], 'n_neighbors': 5}, 
                'regression': [
                    # Option 1: List of target columns (auto-detects predictors)
                    ['loca_road_lanes', 'loca_max_speed'], 
                    # Option 2: List of dict configs
                    [{'target_col': 'loca_road_count', 'predictor_cols': ['col_1',...,'col_n']}]
                ]
            }
        qualitative: Dict with columns
            e.g. { by_category: { 'col_1': 0, ... },
                   by_mode: [ 'col_1', ... ]  }
    
    Returns:
        Pipeline: Complete imputation pipeline
    """
    steps = []
    
    # 1. qualitative imputation (runs first - safer for categoricals)
    if qualitative:
        qual_imputer = create_qualitative_imputing_step(
            by_category=qualitative[ 'by_category' ],
            by_mode=qualitative[ 'by_mode' ]
        )
        
        if len(qual_imputer.steps) > 0:
            # ONLY add if there are steps
            steps.append(
                ( 'qualitative_impute',
                  qual_imputer )
            )
    
    # 2. quantitative imputation (runs after - uses imputed categoricals if needed)
    if quantitative:
        knn = quantitative.get( 'knn',
                                None )
        regression = quantitative.get( 'regression',
                                       None )
        
        quant_imputer = create_quantitative_imputing_step( knn=knn,
                                                           regression=regression )
        
        if len(quant_imputer.steps) > 0:
            # ONLY add if there are steps
            steps.append(
                ( 'quantitative_impute',
                  quant_imputer )
            )
    
    return Pipeline( steps )


# --------
# encoding
# --------
def create_cyclic_encoding_step(
    columns: Optional[List[Tuple[str, int]]]=None, 
    inplace: bool=False
) -> ColumnTransformer:
    transformers = []
    
    if columns:
        transformers.append(
            # cyclical encoding (time/month/hour → sin/cos)
            ( 'cyclic', 
              CyclicEncoding(columns=columns,
                             inplace=inplace)
            )
        )
        
    return ColumnTransformer( transformers,
                              remainder='passthrough',
                              verbose_feature_names_out=False )


def create_ordinal_encoding_step(
    columns: Optional[List[str]]=None, 
    inplace: bool=False
) -> ColumnTransformer:
    transformers = []
    
    if columns:
        transformers.append(
            # ordinal encoding (ordered categories)
            ( 'ordinal', 
              OrdinalEncoding(columns=columns,
                              inplace=inplace)
            )
        )
    
    return ColumnTransformer( transformers,
                              remainder='passthrough',
                              verbose_feature_names_out=False )


def create_ohe_encoding_step(
    columns: Optional[List[str]]=None,
    inplace: bool=False,
    drop_first: bool=False,
    handle_unknown: str='ignore'
) -> ColumnTransformer:
    transformers = []
    
    if columns:
        transformers.append(
            # one-hot encoding (low-cardinality nominals)
            ( 'ohe',
              OneHotEncoding(columns=columns,
                             inplace=inplace,
                             drop_first=drop_first,
                             handle_unknown=handle_unknown)
            )
        )
    
    return ColumnTransformer( transformers,
                              remainder='passthrough',
                              verbose_feature_names_out=False )


def create_target_impact_encoding_step(
    columns: Optional[List[str]]=None,
    target_col: str='',
    smoothing: float=1.0,
    min_samples_leaf: int=1,
    inplace: bool=False
)-> ColumnTransformer:
    transformers = []
    
    if columns:
        transformers.append(
            # target/impact encoding (very high-cardinality)
            ( 'target_impact',
              TargetImpactEncoding(columns=columns,
                                   target_col=target_col,
                                   smoothing=smoothing,
                                   min_samples_leaf=min_samples_leaf,
                                   inplace=inplace),
              columns
            )
        )
    
    return ColumnTransformer( transformers,
                              remainder='passthrough',
                              verbose_feature_names_out=False )


def create_encoding_pipeline(
    ohe_encoding: Optional[Dict]=None,
    ordinal_encoding: Optional[List[str]]=None,
    cyclic_encoding: Optional[List[Tuple[str, int]]]=None, # (col, period)
    target_encoding: Optional[Dict]=None,
    target_impact_encoding: Optional[Dict]=None,
    inplace: bool=False
) -> ColumnTransformer:
    """
    Factory function to create categorical encoding pipeline.
    
    Args:
        ohe_encoding: Columns for one-hot encoding (low-cardinality nominals)
                    { :columns, :drop_first, handle_unknown: :ignore }
        ordinal_encoding: Columns for ordinal encoding (ordered categories)
        cyclic_encoding: Columns for cyclical encoding (sin/cos transformation)
        target_encoding: Columns for target encoding (high-cardinality)
                    { :columns, target_col: :ind_severity }
        target_impact_encoding: columns for very high-cardinality
                    { :columns target_col: :ind_severity, smoothing: 1.0, min_samples_leaf: 1 }
    
    Returns:
        ColumnTransformer: Complete encoding pipeline
    """
    transformers = []
    
    # OHE
    if ohe_encoding:
        columns = ohe_encoding.get( 'columns',
                                    [] )
        drop_first = ohe_encoding.get( 'drop_first',
                                       False )
        handle_unknown = ohe_encoding.get( 'handle_unknown',
                                           'ignore' )
        
        # transformers.append(
        #     # ('name', transformer, columns)
        #     ( 'one_hot_encoding',
        #       OneHotEncoding(columns=columns,
        #                      drop_first=drop_first,
        #                      handle_unknown=handle_unknown,
        #                      inplace=inplace),
        #       columns )
        # )
        transformers.append(
            # ('name', transformer, columns)
            (
                "OHE",
                OneHotEncoder(
                    drop="first" if drop_first else None,
                    handle_unknown=handle_unknown,
                    sparse_output=False
                ),
                columns,
            )
        )
    
    # ordinal encoding
    if ordinal_encoding:
        transformers.append(
            ( 'ordinal_encoding',
              OrdinalEncoding(columns=ordinal_encoding,
                              inplace=inplace),
              ordinal_encoding )
        )
    
    # cyclic encoding
    if cyclic_encoding:
        # cyclic_encoding is List[Tuple[str, int]] → extract just column names
        cyclic_cols = [ col for col, period in cyclic_encoding ]
        
        transformers.append(
            # ('name', transformer, columns)
            ( 'cyclic_encoding',
              CyclicEncoding(columns=cyclic_encoding,
                             inplace=inplace),
              cyclic_cols )
        )
    
    # target encoding
    if target_encoding:
        columns = target_encoding.get( 'columns',
                                       [] )
        target_col = target_encoding.get( 'target_col',
                                          'ind_severity' )
        
        transformers.append(
            # ('name', transformer, columns)
            ( 'target_encoding',
              TargetEncoding(columns=columns,
                             target_col=target_col,
                             inplace=inplace),
              columns )
        )
    
    # target/impact encoding
    if target_impact_encoding:
        columns = target_impact_encoding.get( 'columns',
                                              [] )
        target_col = target_impact_encoding.get( 'target_col'
                                                 'ind_severity' )
        smoothing = target_impact_encoding.get( 'smoothing',
                                                1.0 )
        min_samples_leaf = target_impact_encoding.get( 'min_samples_leaf',
                                                       1 )
        
        # skip if no columns or no target_col
        if columns and target_col:
            transformers.append(
                # ('name', transformer, columns)
                ( 'target_impact_encoding',
                  TargetImpactEncoding(columns=columns,
                                       target_col=target_col,
                                       smoothing=smoothing,
                                       min_samples_leaf=min_samples_leaf,
                                       inplace=inplace),
                  columns )
            )
    
    return ColumnTransformer( transformers,
                              remainder='passthrough',
                              verbose_feature_names_out=False )


# --------
# scaling
# --------
def create_scaling_pipeline(
    minmax_cols: Optional[List[str]]=None,
    robust_cols: Optional[List[str]]=None,
    standard_cols: Optional[List[str]]=None
) -> Pipeline:
    """
    Factory function to create scaling pipeline.

    Args:
        minmax_cols (List[str]): columns for MinMaxScaler (0-1 normalization)
        robust_cols (List[str]): columns for RobustScaler (outlier-resistant)
        standard_cols (List[str]): columns for StandardScaler (mean=0, std=1)

    Returns:
        [Pipeline]: scaling pipeline
    """
    steps = []
    
    # standardization
    if standard_cols:
        steps.append(
            ( 'standard_scaling', 
              Standardization(columns=standard_cols) )
        )
    
    # nominalization via minmax scaling
    if minmax_cols:
        steps.append(
            ( 'minmax_scaling', 
              NominalizationMinMax(columns=minmax_cols) )
        )
    
    # nominalization via robust scaling
    if robust_cols:
        steps.append(
            ( 'robust_scaling', 
              NominalizationRobust(columns=robust_cols) )
        )
    
    scaling_pipeline = Pipeline( steps )
    
    return scaling_pipeline


# -------------------------------
# HIGH-LEVEL (ABSTRACT) PIPELINES
# -------------------------------
def create_data_cleaning_pipeline(
    irrelevant_columns: Optional[List[str]]=None,
    inplace: bool=False
) -> Pipeline:
    """Factory function to create cleaning pipeline
    
    UC:
        to handle
            - type fixes, parsing dates
            - invalid values, outliers, NaNs
        operations that do NOT depend on modeling choices
    
    Args:
        irrelevant_columns (List[str]): columns to drop

    Returns:
        Pipeline: cleaning pipeline
    """
    steps = []
    
    if irrelevant_columns:
        # (name, transformer, columns_to_pass_into_transformer)
        steps.append(
            ( 'drop_irrelevant_columns', 
              DropIrrelevantColumns(columns=irrelevant_columns,
                                    inplace=inplace) )
        )        
    
    return Pipeline( steps )


def create_data_preprocessing_pipeline(
    reducing_config: Dict=None,
    binning_config: Dict=None,
    imputing_config: Dict=None,
    scaling_config: Dict=None,
    encoding_config: Dict=None,
    inplace: bool=False
) -> Pipeline:
    """
    Factory function to create data preparation pipeline.
    
    Args:
        cleaning_config = { irrelevant_columns: [] }
        
        reducing_config = { mapping_dict: {},
                            suffix: str,
                            drop_original: bool }
        
        binning_config = { binning_config: { config1: { columns: [],
                                                        bins: [],
                                                        labels: [] },
                                            config2: { columns: [],
                                                        bins: [],
                                                        labels: [] } }
                           suffix: str } }
        
        imputing_config = { quantitative: { knn: { columns: [], n_neighbors: 5},
                                            regression: { target_col: str, predictor_cols: []} },
                            qualitative: { col1: 0, col2: 4, col3: 5 }
        
        scaling_config = { standardization: { standard: [] },
                           nominalization: { minmax: [],
                                            robust: [] } }
        
        encoding_config = { ohe: { columns: [], drop_first: bool },
                            ordinal: [],
                            cyclic: [('col', period)],
                            target: { columns: [], target_col: str },
                            target_impact: { columns: [], target_col: str, smoothing: float } }
    
    UC:
    ---
        - feature engineering (bins, reducing categories)
        - mappings
        - encoding
        - scaling
        - anything to tune / change

    Steps:
    -----
        - Regroup high-cardinality categoricals (RegroupCategoricalColumns)
        - Encode 'acc_municipality' as integer ID via factorize
    """
    # 1 reduce qualitative modalities
    # reducing_config = { mapping_dict: {},
    #                     suffix: str,
    #                     drop_original: bool }
    reducing_pipe = create_reducing_step(
        mapping_dicts=reducing_config['mapping_dict'],
        suffix=reducing_config['suffix'],
        drop_original=reducing_config['drop_original']
    )
    
    # 2 bin quantitative to qualitative ordinal
    # binning_config = { binning_config: { config1: { columns: [],
    #                                                 bins: [],
    #                                                 labels: [] },
    #                                      config2: { columns: [],
    #                                                 bins: [],
    #                                                 labels: [] } }
    #                    suffix: str } }
    binning_pipe = create_binning_step(
        binning_config=binning_config['binning_config'],
        suffix=binning_config['suffix'],
        inplace=inplace
    )
    
    # 3 imputation
    # imputing_config = { quantitative: { knn: {}, regression: {} },
    #                     qualitative: { by_category: {}, by_mode: [] }
    imputing_pipe = create_imputing_pipeline(
        quantitative=imputing_config['quantitative'],
        qualitative=imputing_config['qualitative']
    )
    
    # 4 scaling
    # scaling_config = { 
    #    standardization: { standard: [] },
    #    nominalization: { minmax: [],
    #                      robust: [] }
    # }
    scaling_pipe = create_scaling_pipeline(
        minmax_cols=scaling_config['nominalization']['minmax'],
        robust_cols=scaling_config['nominalization']['robust'],
        standard_cols=scaling_config['standardization']['standard']
    )
    
    # 5 encoding (ColumnTransformers)
    # encoding_config = { ohe: {},
    #                     ordinal: [],
    #                     cyclic: [],
    #                     target: {},
    #                     target_impact: {}
    # }
    encoding_pipe = create_encoding_pipeline(
        ohe_encoding=encoding_config['ohe'],
        ordinal_encoding=encoding_config['ordinal'],
        cyclic_encoding=encoding_config['cyclic'],
        target_encoding=encoding_config['target'],
        target_impact_encoding=encoding_config['target_impact'],
        inplace=inplace
    )
    
    preparation_pipeline = [
        ( 'regrouping',
          RegroupCategoricalColumns() ),
        
        ( 'reducing',
          reducing_pipe ),
        
        ( 'factorize',
          EncodeMunicipalityAsId(column='acc_municipality') ),
        
        ( 'imputing',
          imputing_pipe ),
        
        ( 'nan_check',
          NaNCheckTransformer('after_imputing') ),
        
        ( 'binning',
          binning_pipe ),
        
        ( 'scaling',
          scaling_pipe ),
        
        ( 'encoding',
          encoding_pipe )
    ]
    
    return Pipeline( preparation_pipeline )


def create_full_pipeline(
    cleaning_config: Dict=None,     # data cleaning params
    reducing_config: Dict=None,     # reducing params
    binning_config: Dict=None,      # binning params
    imputing_config: Dict=None,     # imputing params
    scaling_config: Dict=None,      # scaling params
    encoding_config: Dict=None,     # encoding params
    inplace: bool=False             # global params
) -> Pipeline:
    """
    Factory function to create COMPLETE ML preprocessing pipeline.
    
    Args:
    cleaning_config = { irrelevant_columns: [] }
    reducing_config = { mapping_dict: {},
                        suffix: str,
                        drop_original: bool }
    binning_config = { columns: [],
                       bins: [],
                       labels: [],
                       suffix: str }
    imputing_config = { quantitative: { knn: { columns: [], n_neighbors: 5},
                                        regression: { target_col: str, predictor_cols: []} },
                        qualitative: { 'col1': 0, 'col2': 4, 'col3': 5 }
    scaling_config = { 
        standardization: { standard: [] },
        nominalization: { minmax: [],
                          robust: [] } }
    encoding_config = { ohe: { columns: [], drop_first: 'first', handle_unknown: 'ignore' },
                        ordinal: [],
                        cyclic: [('col', period)],
                        target: { columns: [], target_col: str },
                        target_impact: { columns: [], target_col: str, smoothing: 1.0, min_samples_leaf: 1 } }
    
    Returns:
        Pipeline: Complete end-to-end preprocessing pipeline
    """
    # 1. data cleaning
    cleaning_pipe = create_data_cleaning_pipeline(
        irrelevant_columns=cleaning_config['irrelevant_columns'],
        inplace=inplace
    )
    
    # 2. data preparation
    prep_pipe = create_data_preprocessing_pipeline(
        reducing_config=reducing_config,
        binning_config=binning_config,
        imputing_config=imputing_config,
        scaling_config=scaling_config,
        encoding_config=encoding_config,
        inplace=inplace
    )
    
    # 3. full pipeline orchestration
    full_pipeline = Pipeline([
        ( 'cleaning',
          cleaning_pipe ),
        ( 'preparation',
          prep_pipe )
    ])
    
    return full_pipeline


# ---------------
# HELPER
# ---------------
def get_feature_names_from_pipeline(
    pipeline,
    prep_step_name="preparation",
    enc_step_name="encoding"
):
    preprocessor = pipeline.named_steps[prep_step_name]
    encoding = preprocessor.named_steps[enc_step_name]
    
    return encoding.get_feature_names_out()


# def get_full_feature_names_from_preprocessor(
#     X: pd.DataFrame,
#     preprocessor: Pipeline
# ) -> List[str]:
#     """
#     Get full feature names from a fitted preprocessing Pipeline.
#     Only the final ColumnTransformer's names are trusted;
#     earlier steps rely on DataFrame column names.
#     """
#     X_work = X.copy()

#     for name, step in preprocessor.named_steps.items():
#         # Final encoding ColumnTransformer: trust its feature names
#         if isinstance(step, ColumnTransformer):
#             # Transform to numpy and rebuild DataFrame with proper names
#             out = step.transform(X_work)
#             try:
#                 feature_names = step.get_feature_names_out()
#             except TypeError:
#                 feature_names = step.get_feature_names_out(X_work.columns)
#             X_work = pd.DataFrame(out, columns=feature_names, index=X_work.index)

#         else:
#             # Any other step: just transform and keep DataFrame column names
#             X_work = step.transform(X_work)
#             # If the step returns numpy array, wrap back into DataFrame without renaming
#             if not isinstance(X_work, pd.DataFrame):
#                 X_work = pd.DataFrame(X_work, index=X.index)

#     return list(X_work.columns)

def get_full_feature_names_from_preprocessor(
    X: pd.DataFrame,
    preprocessor: Pipeline
) -> List[str]:
    """
    Get full feature names from a fitted preprocessing Pipeline.
    Trust the final ColumnTransformer's names; earlier steps
    just propagate DataFrame column names.
    """
    X_work = X.copy()

    for name, step in preprocessor.named_steps.items():
        if isinstance(step, ColumnTransformer):
            out = step.transform(X_work)
            try:
                feature_names = step.get_feature_names_out()
            except TypeError:
                feature_names = step.get_feature_names_out(X_work.columns)
            X_work = pd.DataFrame(out, columns=feature_names, index=X_work.index)
        else:
            X_work = step.transform(X_work)
            if not isinstance(X_work, pd.DataFrame):
                X_work = pd.DataFrame(X_work, index=X.index)

    return list(X_work.columns)


# ----------------------
# READY-TO-USE PIPELINE
# ----------------------
def build_default_full_pipeline() -> Pipeline:
    """
    Build a full pre-processing pipeline with project defaults.
    Use this when you just want a ready-to-use pipeline without
    manually specifying all parameters.
    
    For custom behaviour, call method directly.
    
    Takes no arguments.

    Returns:
        Pipeline: returns fully created default pipeline
    """
    # global
    inplace = True
    
    # cleaning
    cleaning_config = { 'irrelevant_columns': IRRELEVANT_COLUMNS }
    
    # reducing
    reducing_config = {
        'mapping_dict': QUALITATIVE_REDUCING_MODALITIES_DICT,
        'suffix': None,
        'drop_original': False
    }
    
    # binning
    binning_config = {
        'binning_config': {
            'lanes': {
                'columns': QUANTITATIVE_TO_QUALITATIVE_ORDINAL['lanes'],
                'bins': [1, 3, 5, float('inf')],
                'labels': [1, 2, 3]
            },
            'speed': {
                'columns': QUANTITATIVE_TO_QUALITATIVE_ORDINAL['speed'],
                'bins': [0, 50, 90, float('inf')],
                'labels': [1, 2, 3]
            }
        },
        'suffix': '_ord'
    }
    
    # imputing config
    imputing_config = {
        'quantitative': {
            'knn': {
                'columns': QUANTITATIVE_COLUMNS,
                'n_neighbors': 5
            },
            # 2 options are available: choose below either
            # list of dicts: [{'target_col': str, 'predictor_cols': ['col1', ...]}, ...]
            # list of target cols: ['col1', ...] (auto-detects predictors)
            'regression': QUANTITATIVE_COLUMNS
        },
        # qualitative = { by_category: { col1: 0, ... }, by_mode: ['col1', ...] }
        'qualitative': {
            'by_category': CATEGORICAL_IMPUTE_VALUE,
            'by_mode': CATEGORICAL_MODE_IMPUTE_COLUMNS
        }
    }
    
    # scaling config
    scaling_config = {
        'standardization': {
            'standard': QUANTITATIVE_SCALING_COLUMNS['standardization']
        },
        'nominalization': {
            'minmax': QUANTITATIVE_SCALING_COLUMNS['minmax_scaler'],
            'robust': QUANTITATIVE_SCALING_COLUMNS['robust_scaler']
        }
    }
    
    # encoding config
    encoding_config = {
        'ohe': {
            'columns': ONE_HOT_ENCODER_COLUMNS,
            'drop_first': 'first',
            'handle_unknown': 'ignore'
        },
        'ordinal': ORDINAL_COLUMNS,
        'cyclic': CYCLIC_COLUMNS,
        'target': {
            'columns': TARGET_ENCODER_COLUMNS,
            'target_col': None
        },
        'target_impact': {
            'columns': TARGET_IMPACT_ENCODER_COLUMNS,
            'target_col': 'ind_severity',
            'smoothing': 1.0,
            'min_samples_leaf': 1
        }
    }
    
    pipeline = create_full_pipeline(
        cleaning_config=cleaning_config,
        reducing_config=reducing_config,
        binning_config=binning_config,
        imputing_config=imputing_config,
        scaling_config=scaling_config,
        encoding_config=encoding_config,
        inplace=inplace
    )
    
    return pipeline
