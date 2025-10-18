import pandas as pd
import numpy as np
from src.utils import to_json_serializable

def get_summary(df: pd.DataFrame) -> dict:
    try:
        desc = (
            df.describe(include="all")
            .round(4)
            .replace([np.nan, np.inf, -np.inf], "n/a")
            .fillna("n/a")
        )

        # Convert dataframe to a JSON-safe dictionary
        return desc.to_dict()
    except Exception as e:
        return {"error": str(e)}