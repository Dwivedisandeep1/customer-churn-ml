import numpy as np
import pandas as pd


def create_features(df):
    """
    Create production features required by the churn prediction model.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw customer data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered features.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Total number of subscribed services
    # --------------------------------------------------
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        df[service_columns]
        .apply(lambda row: (row == "Yes").sum(), axis=1)
    )

    # --------------------------------------------------
    # 2. Convert TotalCharges to numeric
    # --------------------------------------------------
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Missing TotalCharges means no charges yet
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --------------------------------------------------
    # 3. Average monthly revenue
    # --------------------------------------------------
    df["AvgMonthlyRevenue"] = (
        df["TotalCharges"]
        / df["tenure"].replace(0, np.nan)
    )

    # For new customers, use current MonthlyCharges
    df["AvgMonthlyRevenue"] = (
        df["AvgMonthlyRevenue"]
        .fillna(df["MonthlyCharges"])
    )

    # --------------------------------------------------
    # 4. Customer tenure groups
    # --------------------------------------------------
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    return df