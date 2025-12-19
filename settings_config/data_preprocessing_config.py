"""
author: Michael Munz

All constants are collected into a single, easily-swappable file.
To try different scnearios, do swapping the file without touching pipeline code.

"""
from typing import Dict, List, Tuple

# ----------------------------------------
# MODULE-LEVEL CONSTANTS (FOR REUSABILITY)
# ----------------------------------------
IRRELEVANT_COLUMNS: List[str] = [
    # accident-level
    "acc_date",
    "acc_department",
    "acc_long",
    "acc_lat",
    "acc_metro",
    
    # individual-level
    "ind_action",
    "ind_age",
    "ind_location",
    "ind_secu2",
    "ind_year",
    
    # vehicle-level
    "veh_id",
]

CATEGORICAL_COLUMNS: List[str] = [
    # accident-level context
    "acc_municipality",
    "acc_ambient_lightning",
    "acc_urbanization_level",
    "acc_intersection",
    "acc_atmosphere",
    "acc_collision_type",
    
    # individual-level context
    "ind_place",
    "ind_cat",
    "ind_sex",
    "ind_trip",
    "ind_secu1",
    "ind_age_group",
    
    # location-level context
    "loca_road_cat",
    "loca_traffic_circul",
    "loca_road_gradient",
    "loca_road_view",
    "loca_road_surface_cond",
    "loca_accident",
    "loca_is_intersection",
    
    # vehicle-level context
    "veh_cat",
    "veh_fixed_obstacle",
    "veh_moving_obstacle",
    "veh_impact",
    "veh_maneuver",
    "veh_motor"
]

CATEGORICAL_IMPUTE_VALUE: Dict[str, int] = {
    # 0 = 'unknown'
    "loca_traffic_circul": 0,
    "loca_road_gradient": 0,
    "loca_road_view": 0,
    "loca_road_surface_cond": 0,
    "loca_accident": 0,
    "veh_moving_obstacle": 0,
    "veh_impact": 0,
    "veh_fixed_obstacle": 0,
    "veh_maneuver": 0,
    "veh_motor": 0,
    
    # 4 = 'unknown'
    "acc_atmosphere": 4,
    "acc_collision_type": 4,
    "ind_secu1": 4,
    
    # 5 = 'unknown'
    "acc_intersection": 5,
    "ind_trip": 5,
    
    # 6 = 'unknown'
    "veh_cat": 6,
    
    # 8 = 'unknown'
    "loca_road_cat": 8,
}

CATEGORICAL_MODE_IMPUTE_COLUMNS = [
    # accident-level
    "acc_ambient_lightning",
    
    # individual-level
    "ind_sex"
]

EXCLUDED_FROM_REGRESSION = [
    # any datetime / non-numeric can be explicitly blacklisted
    
    # accident-level
    "acc_date"
]

ORDINAL_COLUMNS: List[str] = [
    # individual-level
    "ind_age_group",
    
    # location-level
    "loca_road_cat",
    "loca_road_surface_cond",
    "loca_road_lanes_ord",
    "loca_max_speed_ord",
]

NOMINAL_COLUMNS: List[str] = [
    # accident-level
    "acc_municipality",
    "acc_ambient_lightning",
    "acc_urbanization_level",
    "acc_intersection",
    "acc_atmosphere",
    "acc_collision_type",
    
    # individual-level
    "ind_place",
    "ind_cat",
    "ind_sex",
    "ind_secu1",
    
    # location-level
    "loca_traffic_circul",
    "loca_road_gradient",
    "loca_road_view",
    "loca_accident",
    "loca_is_intersection",
    
    # vehicle-level
    "veh_cat",
    "veh_fixed_obstacle",
    "veh_moving_obstacle",
    "veh_impact",
    "veh_maneuver",
    "veh_motor",
]

QUANTITATIVE_COLUMNS: List[str] = [
    # accident-level
    "acc_year",
    
    # location-level
    "loca_road_lanes",
    "loca_max_speed",
    "loca_max_speed_dif",
]

QUANTITATIVE_SCALING_COLUMNS: Dict[str, List[str]] = {
    # standardization (z-scorer)
    "standardization": ["acc_year"],
    
    # normalization
    "minmax_scaler": [
        "loca_road_count",
        "loca_max_speed",
        "loca_max_speed_dif"
    ],
    "robust_scaler": [ "loca_road_lanes" ],
}

QUANTITATIVE_TO_QUALITATIVE_ORDINAL: Dict[str, str] = {
    # location-level
    "lanes": [ "loca_road_lanes" ],
    "speed": [ "loca_max_speed" ],
}

CYCLIC_COLUMNS: List[Tuple[str, int]] = [
    # accident-level
    ("acc_month", 12),
    ("acc_hour", 24),
]

TARGET_ENCODER_COLUMNS: List[str] = []

TARGET_IMPACT_ENCODER_COLUMNS: List[str] = [
    # accident-level
    "acc_municipality",
]

ONE_HOT_ENCODER_COLUMNS: List[str] = [
    # accident-level
    "acc_ambient_lightning",
    "acc_urbanization_level",
    "acc_intersection",
    "acc_atmosphere",
    "acc_collision_type",
    
    # individual-level
    "ind_place",
    "ind_cat",
    "ind_sex",
    "ind_trip",
    "ind_secu1",
    
    # location-level
    "loca_traffic_circul",
    "loca_road_gradient",
    "loca_road_view",
    "loca_accident",
    
    # vehicle-level
    "veh_cat",
    "veh_fixed_obstacle",
    "veh_moving_obstacle",
    "veh_impact",
    "veh_maneuver",
    "veh_motor",
]

REGROUP_RULES: Dict[str, Dict[int | list[int], int]] = {
    # vehicle-level
    "veh_cat": {
        10: 7, 80: 1,
        2: 33, 31: 33,
        13: 14, 15: 14,
        38: 37, 39: 37, 40: 37,
        16: 17, 20: 17, 21: 17,
        3: 99,
        60: 50,
        32: 30, 34: 30, 35: 30, 36: 30, 41: 30, 42: 30, 43: 30,
    },
    "veh_fixed_obstacle": {
        3: 4,
        7: 5, 9: 5, 10: 5, 11: 5, 12: 5, 14: 5, 15: 5, 16: 5,
    },
    "veh_moving_obstacle": {
        4: 9, 5: 9, 6: 9,
    },
    "veh_maneuver": {
        12: 11,
        14: 13,
        16: 15,
        18: 17,
        4: 98, 10: 98, 20: 98, 22: 98, 24: 98,
        3: 99, 6: 99, 7: 99, 8: 99, 21: 99, 25: 99,
    },
}

QUALITATIVE_REDUCING_MODALITIES_DICT: Dict[str, Dict[int, int]] = {
    # accident-level
    "acc_ambient_lightning": {
        0: 0,
        1: 1,
        2: 2, 5: 2,
        3: 3, 4: 3,
    },
    "acc_urbanization": {
        0: 0,
        1: 1,
        2: 2,
    },
    "acc_intersection": {
        1: 1,
        2: 2, 3: 2, 4: 2,
        6: 3,
        5: 4, 7: 4, 8: 4, 9: 4,
    },
    "acc_atmosphere": {
        1: 1,
        2: 2, 3: 2,
        8: 3,
        4: 4, 5: 4, 6: 4, 7: 4, 9: 4, 0: 4,
    },
    "acc_collision_type": {
        1: 1,
        2: 2,
        3: 3,
        4: 4, 5: 4, 6: 4, 7: 4, 0: 4,
    },
    
    # individual-level
    "ind_place": {
        1: 1,
        **{i: 2 for i in range(2, 10)},
        10: 3,
    },
    "ind_cat": {
        1: 1,
        2: 2,
        3: 3, 4: 3,
    },
    "ind_sex": {
        1: 1,
        2: 2,
    },
    "ind_trip": {
        1: 1, 2: 1,
        3: 2,
        4: 3,
        5: 4,
        0: 5, 9: 5,
    },
    "ind_secu1": {
        0: 0,
        1: 1,
        2: 2,
        3: 3, 4: 3, 9: 3,
        8: 4, -1: 4,
    },
    "ind_age_group": {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    },
    
    # location-level
    "loca_road_cat": {
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 6, 7: 7, 8: 8,
    },
    "loca_traffic_circul": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    },
    "loca_road_gradient": {
        0: 0,
        1: 1, 2: 1,
        3: 2, 4: 2,
    },
    "loca_road_view": {
        0: 0,
        1: 1,
        2: 2, 3: 2,
        4: 3,
    },
    "loca_road_surface_cond": {
        0: 0, 9: 0,
        1: 1, 2: 1,
        3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2,
    },
    "loca_accident": {
        0: 0,
        1: 1, 2: 1,
        3: 2,
        4: 3, 5: 3,
        6: 4, 7: 4,
    },
    
    # vehicle-level
    "veh_cat": {
        7: 1,
        33: 2, 30: 2,
        1: 3,
        14: 4, 37: 4,
        50: 5,
        0: 6, 17: 6, 99: 6,
    },
    "veh_fixed_obstacle": {
        0: 0,
        1: 1, 2: 1, 4: 1, 5: 1, 6: 1,
        8: 2, 13: 2, 17: 2,
    },
    "veh_moving_obstacle": {
        0: 0,
        1: 1,
        2: 2,
        9: 3,
    },
    "veh_impact": {
        0: 0,
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3,
        9: 4,
    },
    "veh_maneuver": {
        0: 0,
        1: 1, 2: 1,
        9: 2, 11: 2, 13: 2,
        15: 3,
        5: 4, 17: 4, 19: 4,
        23: 5,
        26: 6, 98: 6, 99: 6,
    },
    "veh_motor": {
        -1: 0, 0: 0,
        1: 1, 2: 1,
        3: 2, 4: 2, 6: 2,
        5: 3,
    },
}
