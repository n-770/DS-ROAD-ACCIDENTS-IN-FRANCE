import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Available classes:
#------------------------------------
# RemoveIrrelevantCols
# RearrangeCatCols
# ConditionalMultiQuantImputer
# ConditionalCatImputer
# AggrLocaSplit
# SupervisedEncoderWrapper
# RemoveIdCols
# TrigonometricEncoder
# SafeColumnSelector


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class ModalityReducer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to reduce categorical modalities
    using predefined mapping dictionaries.
    """

    def __init__(self, mapping_dicts=None, loca_max_speed_map=None):
        self.mapping_dicts = mapping_dicts if mapping_dicts is not None else {}
        self.loca_max_speed_map = loca_max_speed_map if loca_max_speed_map is not None else {}

    def fit(self, X, y=None):
        # Nothing to learn, just return self
        return self

    def transform(self, X):
        X = X.copy()

        # Apply mappings for categorical variables
        for col, mapping in self.mapping_dicts.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(X[col])

        # Special handling for loca_max_speed
        if "loca_max_speed" in X.columns and self.loca_max_speed_map:
            X["loca_max_speed"] = X["loca_max_speed"].map(self.loca_max_speed_map).fillna(X["loca_max_speed"])

        return X


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class RemoveIrrelevantCols(BaseEstimator, TransformerMixin):
    """
    Transformer to remove unnecessary columns from the road accidents DataFrame.
    If a column does not exist, it will be ignored and a warning printed.
    """

    def __init__(self, cols_to_drop=None, verbose=True):
        # Default columns to drop if none provided
        if cols_to_drop is None:
            cols_to_drop = [
                "veh_id",
                "ind_year",
                "acc_date",
                "ind_age",
                "acc_metro",
                "acc_long",
                "acc_lat",
                "ind_secu2",
                "ind_location",
                "ind_action",
                "acc_department",

                # "acc_intersection",
                # "veh_motor",
                # "acc_ambient_lightning",
                # "acc_atmosphere",

                # "ind_place",
                # "ind_trip",
                # "veh_impact",

                #"acc_hour",
                #"acc_month",
                #"ind_sex",
                #"acc_municipality"
                
            ]
        self.cols_to_drop = cols_to_drop
        self.verbose = verbose
        self._feature_names_out = None

    def fit(self, X, y=None):
        # Store the feature names that remain after dropping
        if isinstance(X, pd.DataFrame):
            self._feature_names_out = [
                col for col in X.columns if col not in self.cols_to_drop
            ]
        return self

    def transform(self, X):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_cols = [col for col in self.cols_to_drop if col not in X.columns]
        if missing_cols and self.verbose:
            print("ℹ️ RemoveIrrelevantCols -> The following columns were not found and skipped:", missing_cols)

        return X.drop(columns=[col for col in self.cols_to_drop if col in X.columns])

    def get_feature_names_out(self, input_features=None):
        # Return stored feature names if available
        if self._feature_names_out is not None:
            return self._feature_names_out
        # Fallback: drop cols from input_features
        if input_features is not None:
            return [col for col in input_features if col not in self.cols_to_drop]
        raise ValueError("Feature names are not available. Fit the transformer first.")


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class RearrangeCatCols(BaseEstimator, TransformerMixin):
    """
    Transformer to regroup classes of categorical variables with high cardinality
    for veh_cat, veh_fixed_obstacle, veh_moving_obstacle, and veh_maneuver.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._feature_names_in = None

    def fit(self, X, y=None):
        # Store input feature names for later use
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
        return self

    def transform(self, X):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        X = X.copy()  # avoid mutating original

        # --- veh_cat ---
        if 'veh_cat' in X.columns:
            X['veh_cat'] = X['veh_cat'].replace({
                10: 7,   # Light vehicle regroup
                80: 1,   # Bicycle/E-Bike
                3: 99,   # Other Vehicle
                60: 50   # EDP
            })
            X['veh_cat'] = X['veh_cat'].replace({2: 33, 31: 33})   # Motorcycle/Moped
            X['veh_cat'] = X['veh_cat'].replace({13: 14, 15: 14})  # Heavy goods vehicle
            X['veh_cat'] = X['veh_cat'].replace({38: 37, 39: 37, 40: 37})  # Public Transport
            X['veh_cat'] = X['veh_cat'].replace({16: 17, 20: 17, 21: 17})  # Tractor
            X['veh_cat'] = X['veh_cat'].replace({32: 30, 34: 30, 35: 30, 36: 30,
                                                41: 30, 42: 30, 43: 30})  # Scooter/3WD/Quad
        elif self.verbose:
            print("ℹ️ Column 'veh_cat' not found, skipping regrouping.")

        # --- veh_fixed_obstacle ---
        if 'veh_fixed_obstacle' in X.columns:
            X['veh_fixed_obstacle'] = X['veh_fixed_obstacle'].replace({
                3: 4,  # Concrete/Metal Barrier
            })
            X['veh_fixed_obstacle'] = X['veh_fixed_obstacle'].replace({
                7: 5, 9: 5, 10: 5, 11: 5, 12: 5, 14: 5, 15: 5, 16: 5
            })  # Other barrier
        elif self.verbose:
            print("ℹ️ Column 'veh_fixed_obstacle' not found, skipping regrouping.")

        # --- veh_moving_obstacle ---
        if 'veh_moving_obstacle' in X.columns:
            X['veh_moving_obstacle'] = X['veh_moving_obstacle'].replace({
                4: 9, 5: 9, 6: 9  # Other
            })
        elif self.verbose:
            print("ℹ️ Column 'veh_moving_obstacle' not found, skipping regrouping.")

        # --- veh_maneuver ---
        if 'veh_maneuver' in X.columns:
            X['veh_maneuver'] = X['veh_maneuver'].replace({
                12: 11,  # Changing lanes left/right
                14: 13,  # Offset left/right
                16: 15,  # Turning left/right
                18: 17   # Overtaking left/right
            })
            X['veh_maneuver'] = X['veh_maneuver'].replace({
                4: 98, 10: 98, 20: 98, 22: 98, 24: 98  # Others less severe
            })
            X['veh_maneuver'] = X['veh_maneuver'].replace({
                3: 99, 6: 99, 7: 99, 8: 99, 21: 99, 25: 99  # Others highly severe
            })
        elif self.verbose:
            print("ℹ️ Column 'veh_maneuver' not found, skipping regrouping.")

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for consistency in pipelines.
        """
        if input_features is not None:
            return input_features
        elif self._feature_names_in is not None:
            return self._feature_names_in
        else:
            raise ValueError("No feature names stored. Fit the transformer first.")

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class ConditionalMultiQuantImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in multiple quantitative variables using the distribution
    conditional on a categorical variable. If conditional distribution is unavailable,
    fall back to the global distribution.
    """

    def __init__(self, quant_vars, cat_var, random_state=42):
        self.quant_vars = quant_vars        # list of quantitative variables
        self.cat_var = cat_var              # conditioning categorical variable
        self.random_state = random_state
        self._feature_names_in = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        rng = np.random.default_rng(self.random_state)

        # Store input feature names
        self._feature_names_in = df.columns.tolist()

        # Store distributions per variable per category
        self.global_vals_ = {}
        self.cond_vals_ = {}

        for q in self.quant_vars:
            self.global_vals_[q] = df[q].dropna().values
            self.cond_vals_[q] = {}
            for cat in df[self.cat_var].dropna().unique():
                cond_vals = df.loc[(df[self.cat_var] == cat) & (~df[q].isna()), q].values
                if len(cond_vals) > 0:
                    self.cond_vals_[q][cat] = cond_vals
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        rng = np.random.default_rng(self.random_state)

        for q in self.quant_vars:
            missing_mask = df[q].isna()
            for idx in df[missing_mask].index:
                cat_value = df.loc[idx, self.cat_var]
                if cat_value in self.cond_vals_[q]:
                    df.loc[idx, q] = rng.choice(self.cond_vals_[q][cat_value])
                else:
                    df.loc[idx, q] = rng.choice(self.global_vals_[q])
        return df

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for consistency in pipelines.
        """
        if input_features is not None:
            return input_features
        elif self._feature_names_in is not None:
            return self._feature_names_in
        else:
            raise ValueError("No feature names stored. Fit the transformer first.")


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class ConditionalCatImputer(BaseEstimator, TransformerMixin):
    """
    Pipeline-compatible transformer that imputes missing categorical values
    using conditional distributions based on the target variable (if provided).

    Parameters
    ----------
    var_list : list
        List of categorical variables to impute.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print warnings about missing columns.
    """

    def __init__(self, var_list, random_state=42, verbose=True):
        self.var_list = var_list
        self.random_state = random_state
        self.verbose = verbose
        self._feature_names_in = None

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        self.rules_ = {}

        # Store input feature names
        self._feature_names_in = X.columns.tolist()

        # Ensure y is a Series aligned with X.index if provided
        if y is not None:
            y = pd.Series(y, index=X.index)

        for feat in self.var_list:
            if feat not in X.columns:
                if self.verbose:
                    print(f"ℹ️ ConditionalCatImputer -> Column '{feat}' not found, skipping.")
                continue

            # Global distribution
            global_probs = X[feat].value_counts(normalize=True, dropna=True)
            self.rules_[feat] = {
                "global_choices": global_probs.index.to_numpy(),
                "global_p": global_probs.values,
                "conditional": {}
            }

            # Conditional distributions per target class
            if y is not None:
                for cls in y.unique():
                    mask = (y == cls)
                    cond_probs = X.loc[mask, feat].value_counts(normalize=True, dropna=True)
                    if not cond_probs.empty:
                        self.rules_[feat]["conditional"][cls] = (
                            cond_probs.index.to_numpy(),
                            cond_probs.values
                        )

        return self

    def transform(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        X = X.copy()

        # Align y with X.index if provided
        if y is not None:
            y = pd.Series(y, index=X.index)

        for feat, feat_rules in self.rules_.items():
            if feat not in X.columns:
                continue

            missing_mask = X[feat].isna()
            if not missing_mask.any():
                continue

            if y is not None:
                # Fill per class in bulk
                for cls, (choices, probs) in feat_rules["conditional"].items():
                    cls_mask = missing_mask & (y == cls)
                    if cls_mask.any():
                        X.loc[cls_mask, feat] = rng.choice(choices, size=cls_mask.sum(), p=probs)

            # Remaining missing → global distribution
            still_missing = X[feat].isna()
            if still_missing.any():
                X.loc[still_missing, feat] = rng.choice(
                    feat_rules["global_choices"],
                    size=still_missing.sum(),
                    p=feat_rules["global_p"]
                )

        return X

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for consistency in pipelines."""
        if input_features is not None:
            return input_features
        elif self._feature_names_in is not None:
            return self._feature_names_in
        else:
            raise ValueError("No feature names stored. Fit the transformer first.")


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class AggrLocaSplit(BaseEstimator, TransformerMixin):
    """
    Custom transformer that aggregates accident-level data by individual and accident identifiers,
    applies one-hot encoding to categorical columns, and returns aggregated features.
    Uses max() for most features, but first() for specific categorical columns.
    """

    def __init__(self, agg_features="max", agg_target="max", keys=None,
                 random_state=42, verbose=True):
        self.agg_features = agg_features
        self.agg_target = agg_target
        self.keys = keys if keys is not None else ['acc_num', 'ind_temp_id']
        self.encoder = None
        self.random_state = random_state
        self.verbose = verbose

        # columns that should use first() instead of max()
        self.cat_cols_first = ["acc_municipality", "acc_department"]

        # internal storage
        self._feature_names_out = None

    def fit(self, X, y=None):
        # Validate keys
        missing_keys = [k for k in self.keys if k not in X.columns]
        if missing_keys:
            raise ValueError(f"Grouping keys not found in X: {missing_keys}")

        # Define categorical columns explicitly
        categorical_cols = [
            'loca_traffic_circul',
            'loca_road_gradient',
            'loca_road_view',
            'loca_accident'
        ]
        categorical_cols = [c for c in categorical_cols if c in X.columns]
        self.categorical_cols_ = categorical_cols

        if self.verbose and not categorical_cols:
            print("ℹ️ AggrLocaSplit -> No categorical columns found for encoding.")

        # Fit encoder
        if categorical_cols:
            self.encoder = OneHotEncoder(
                sparse_output=False,
                drop=None,
                handle_unknown="ignore"
            )
            self.encoder.fit(X[self.categorical_cols_])

        # Optionally aggregate y during fit
        if y is not None:
            grouped_y = (
                pd.DataFrame({'y': y})
                .groupby(X[self.keys].apply(tuple, axis=1))
                .agg(self.agg_target)
            )
            grouped_y.index = pd.MultiIndex.from_tuples(grouped_y.index, names=self.keys)
            self.y_aggr_ = grouped_y.reset_index().rename(columns={'y': 'target'})
        else:
            self.y_aggr_ = None

        # Build feature names out (keys + other + encoded + extra count)
        other_cols = [c for c in X.columns if c not in categorical_cols]
        encoded_names = []
        if self.encoder is not None:
            encoded_names = list(self.encoder.get_feature_names_out(self.categorical_cols_))
        self._feature_names_out = self.keys + other_cols + encoded_names + ["loca_road_count"]

        return self

    def transform(self, X):
        # Validate keys
        missing_keys = [k for k in self.keys if k not in X.columns]
        if missing_keys:
            if self.verbose:
                print(f"ℹ️ No rows for aggregation available (missing keys: {missing_keys}).")
            return X.copy()

        if X.empty:
            if self.verbose:
                print("ℹ️ AggrLocaSplit -> No rows for aggregation available.")
            return X.copy()

        # Check if already aggregated
        if X[self.keys].duplicated().sum() == 0:
            if self.verbose:
                print("ℹ️ AggrLocaSplit -> Data already aggregated, skipping further aggregation.")
            return X.copy()

        categorical_cols = self.categorical_cols_
        other_cols = [c for c in X.columns if c not in categorical_cols]

        # Transform categorical columns
        if categorical_cols:
            encoded = pd.DataFrame(
                self.encoder.transform(X[categorical_cols]),
                columns=self.encoder.get_feature_names_out(categorical_cols),
                index=X.index
            )
            df_combined = pd.concat([X[other_cols], encoded], axis=1)
        else:
            df_combined = X.copy()

        # Aggregation
        grouped_X = df_combined.groupby(self.keys)

        # Build aggregation dict (skip keys)
        agg_dict = {}
        for col in df_combined.columns:
            if col in self.keys:
                continue
            elif col in self.cat_cols_first:
                agg_dict[col] = "first"
            else:
                agg_dict[col] = self.agg_features

        X_aggr = grouped_X.agg(agg_dict)
        X_aggr['loca_road_count'] = grouped_X.size()

        # Ensure categorical-first columns are strings
        for col in self.cat_cols_first:
            if col in X_aggr.columns:
                X_aggr[col] = X_aggr[col].astype(str)

        if self.verbose:
            print(f"ℹ️ AggrLocaSplit -> Aggregated from {len(X)} rows to {len(X_aggr)} groups.")

        return X_aggr.reset_index()

    def transform_y(self, X, y):
        """Aggregate y with the same grouping keys as X."""
        grouped_y = (
            pd.DataFrame({'y': y})
            .groupby(X[self.keys].apply(tuple, axis=1))
            .agg(self.agg_target)
        )
        grouped_y.index = pd.MultiIndex.from_tuples(grouped_y.index, names=self.keys)
        return grouped_y.reset_index()['y']

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for consistency in pipelines.
        """
        if input_features is not None:
            return input_features
        elif self._feature_names_out is not None:
            return self._feature_names_out
        else:
            raise ValueError("No feature names stored. Fit the transformer first.")

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class SupervisedEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Wraps any category_encoders supervised encoder so it can be used inside
    a ColumnTransformer without index mismatch errors.
    Ensures categorical columns are cast to string dtype before encoding.
    """

    def __init__(self, encoder, columns):
        self.encoder = encoder
        self.columns = columns

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.columns, index=y.index if y is not None else None)
        X_df = X_df.astype(str)
        self.encoder.fit(X_df, y)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.columns)
        X_df = X_df.astype(str)
        return self.encoder.transform(X_df)

    def get_feature_names_out(self, input_features=None):
        # Delegate to the underlying encoder if it supports it
        if hasattr(self.encoder, "get_feature_names_out"):
            return self.encoder.get_feature_names_out(input_features)
        # Otherwise, just return the original column names
        return self.columns


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class RemoveIdCols(BaseEstimator, TransformerMixin):
    """
    Transformer to remove unnecessary columns from the road accidents DataFrame.
    If a column does not exist, it will be ignored and a warning printed.
    """

    def __init__(self, cols_to_drop=None, verbose=True):
        # Default columns to drop if none provided
        if cols_to_drop is None:
            cols_to_drop = [
                "ind_temp_id",
                "acc_num",
            ]
        self.cols_to_drop = cols_to_drop
        self.verbose = verbose
        self._feature_names_in = None
        self._feature_names_out = None

    def fit(self, X, y=None):
        # Store input feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            # Precompute output feature names
            self._feature_names_out = [
                col for col in self._feature_names_in if col not in self.cols_to_drop
            ]
        return self

    def transform(self, X):
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_cols = [col for col in self.cols_to_drop if col not in X.columns]
        if missing_cols and self.verbose:
            print("ℹ️ RemoveIdCols -> The following columns were not found and skipped:", missing_cols)

        return X.drop(columns=[col for col in self.cols_to_drop if col in X.columns])

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for consistency in pipelines.
        """
        if input_features is not None:
            return [col for col in input_features if col not in self.cols_to_drop]
        elif self._feature_names_out is not None:
            return self._feature_names_out
        else:
            raise ValueError("No feature names stored. Fit the transformer first.")

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class TrigonometricEncoder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that encodes cyclical (periodic) features
    using sine and cosine transformations.

    This is particularly useful for variables such as time of day, day of week,
    or month of year, where values are inherently cyclical. By mapping them onto
    the unit circle, the encoder preserves continuity between the start and end
    of the cycle (e.g., 23:00 and 00:00 are close in encoded space).

    Parameters
    ----------
    col_periods : dict
        Dictionary mapping column names to their respective periods.
        Example: {"hour": 24, "month": 12}.

    Attributes
    ----------
    new_cols_ : list of str
        Names of the newly created sine and cosine features for each input column.
        Populated during fitting.

    Methods
    -------
    fit(X, y=None)
        Prepares the list of new feature names based on `col_periods`.

    transform(X)
        Returns a DataFrame with sine and cosine encoded features for each
        specified column.

    get_feature_names_out(input_features=None)
        Returns the names of the generated features as a NumPy array.
    """

    def __init__(self, col_periods):
        self.col_periods = col_periods
        self.new_cols_ = []

    def fit(self, X, y=None):
        self.new_cols_ = []
        for col in self.col_periods.keys():
            self.new_cols_.extend([f"{col}_sin", f"{col}_cos"])
        return self

    def transform(self, X):
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        for col, period in self.col_periods.items():
            out[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / period)
            out[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / period)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.new_cols_)


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class SafeColumnSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that safely selects a subset of columns
    from a DataFrame, ensuring only existing columns are retained.

    This is useful in modular pipelines where schema drift or missing columns
    may occur, and you want to avoid errors by automatically filtering out unavailable columns.

    Parameters
    ----------
    candidate_cols : list of str
        List of column names that are desired for selection. Only those present
        in the input DataFrame at fit time will be retained.

    Attributes
    ----------
    valid_cols_ : list of str
        Subset of `candidate_cols` that were found in the input DataFrame during
        fitting. These are the columns actually used in transformation.

    Methods
    -------
    fit(X, y=None)
        Identifies which candidate columns exist in `X` and stores them in
        `valid_cols_`.

    transform(X)
        Returns a DataFrame containing only the `valid_cols_`.

    get_feature_names_out(input_features=None)
        Returns the names of the selected features as a NumPy array.
    """
    
    def __init__(self, candidate_cols):
        self.candidate_cols = candidate_cols
        self.valid_cols_ = []

    def fit(self, X, y=None):
        # Store only those columns that exist at this stage
        self.valid_cols_ = [c for c in self.candidate_cols if c in X.columns]
        return self

    def transform(self, X):
        return X[self.valid_cols_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.valid_cols_)

