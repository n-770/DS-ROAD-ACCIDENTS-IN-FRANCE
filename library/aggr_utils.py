import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def aggr_loca(pa_df):
    """
    Aggregate loca-related columns by acc_num and ind_temp_id.
    Encodes categorical columns with OneHotEncoder, aggregates others with max(),
    and adds loca_road_count (row count per group).
    
    Parameters:
        pa_df (pd.DataFrame): Input DataFrame containing accident data.
    
    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
        
    # Define categorical columns to encode
    categorical_cols = [
        #'loca_road_cat',
        'loca_traffic_circul',
        'loca_road_gradient',
        'loca_road_view',
        #'loca_road_surface_cond',
        'loca_accident'
    ]
    
    # All other columns (numerical + datetime) are treated the same
    other_cols = [col for col in pa_df.columns if col not in categorical_cols]
    
    # Apply OneHotEncoding to categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_array = encoder.fit_transform(pa_df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=pa_df.index
    )
    
    # Combine encoded categorical columns with the rest
    df_combined = pd.concat([pa_df[other_cols], encoded_df], axis=1)
    
    # Group by acc_num and ind_temp_id
    grouped = df_combined.groupby(['acc_num', 'ind_temp_id'])
    
    # Aggregate with max()
    df_aggr = grouped.agg('max')
    
    # Add loca_road_count: count of rows per acc_num/ind_temp_id
    df_aggr['loca_road_count'] = grouped.size()
    
    # Reset index for clean DataFrame
    df_aggr = df_aggr.reset_index()
    
    return df_aggr


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def aggr_loca_split(
    df: pd.DataFrame,
    y: pd.Series,
    encoder: OneHotEncoder | None = None,
    agg_features: str = "max",
    agg_target: str = "max"
):
    """
    Aggregate accident-level data by individual and accident identifiers to aggregate intersection data with multiple location to one row,
    applying one-hot encoding to selected categorical columns and combining
    them with numeric features using max() for aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame containing accident, individual, vehicle, and location details.
    y : pd.Series
        Target variable aligned with df rows (e.g., accident severity).
    encoder : OneHotEncoder, optional
        A scikit-learn OneHotEncoder instance. If None, a new encoder is created.
        Regardless of whether an encoder is passed, it is always refitted on the
        current set of categorical columns to avoid leakage from previously fitted
        categories.
    agg_features : str, default="max"
        Aggregation function applied to feature columns when grouping by keys.
    agg_target : str, default="max"
        Aggregation function applied to the target variable when grouping by keys.

    Returns
    -------
    X_out : pd.DataFrame
        Aggregated feature matrix with one-hot encoded categorical variables and
        numeric features, grouped by accident and individual identifiers.
    y_out : pd.Series
        Aggregated target variable aligned with X_out.
    encoder : OneHotEncoder
        The fitted encoder, refitted on the categorical columns used in this call.

    Notes
    -----
    - Keys used for grouping are ['acc_num', 'ind_temp_id'].
    - A new feature `loca_road_count` is added, representing the number of
      location records per group.
    - Excluded categorical columns (e.g., 'loca_road_cat') will not appear in
      the encoded output unless explicitly listed in `categorical_cols`.
    """

    # Define categorical columns explicitly
    categorical_cols = [
        'loca_traffic_circul',
        'loca_road_gradient',
        'loca_road_view',
        'loca_accident'
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in categorical_cols]

    # Always ensure encoder is fitted only on current categorical_cols
    if encoder is None:
        # Create and fit a fresh encoder
        encoder = OneHotEncoder(
            sparse_output=False,
            drop=None,
            handle_unknown="ignore"
        )
        encoder.fit(df[categorical_cols])
    else:
        # Check if encoder was fitted on the same columns
        if not hasattr(encoder, "feature_names_in_") or \
           set(encoder.feature_names_in_) != set(categorical_cols):
            # Refit only if mismatch
            encoder.fit(df[categorical_cols])

    # Transform categorical columns
    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # Combine with other columns
    df_combined = pd.concat([df[other_cols], encoded], axis=1)

    # Aggregation
    keys = ['acc_num', 'ind_temp_id']
    grouped_X = df_combined.groupby(keys)
    X_aggr = grouped_X.agg(agg_features)
    X_aggr['loca_road_count'] = grouped_X.size()

    grouped_y = pd.DataFrame({'y': y}).groupby(df[keys].apply(tuple, axis=1)).agg(agg_target)
    grouped_y.index = pd.MultiIndex.from_tuples(grouped_y.index, names=keys)

    # Align X and y
    X_aggr = X_aggr.reset_index()
    y_aggr = grouped_y.reset_index().rename(columns={'y': 'target'})
    Xy = X_aggr.merge(y_aggr, on=keys, how='inner')

    y_out = Xy.pop('target')
    X_out = Xy

    return X_out, y_out, encoder
