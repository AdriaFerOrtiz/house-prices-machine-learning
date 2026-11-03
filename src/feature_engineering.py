
import pandas as pd


def add_engineered_features(
    X: pd.DataFrame,
    include_interactions: bool = True
) -> pd.DataFrame:
    """
    Add domain-based engineered features to the Ames Housing dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Original predictors, without Id or SalePrice.
    include_interactions : bool
        Whether to include interaction features.

    Returns
    -------
    pd.DataFrame
        Copy of X with additional engineered features.
    """

    X_fe = X.copy()

    required_columns = [
        "GrLivArea",
        "TotalBsmtSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "YrSold",
        "YearBuilt",
        "YearRemodAdd",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "OverallQual"
    ]

    missing_columns = set(required_columns) - set(X_fe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 1. Total living area: above-ground living area + basement.
    # Missing basement values are treated as zero when the basement is absent.
    X_fe["TotalAreaIncludingBasement"] = (
        X_fe["GrLivArea"] +
        X_fe["TotalBsmtSF"].fillna(0)
    )

    # 2. Total bathrooms.
    # Missing basement bathroom values are treated as zero.
    X_fe["TotalBathrooms"] = (
        X_fe["FullBath"] +
        0.5 * X_fe["HalfBath"] +
        X_fe["BsmtFullBath"].fillna(0) +
        0.5 * X_fe["BsmtHalfBath"].fillna(0)
    )

    # 3. Total finished basement area.
    X_fe["TotalFinishedBsmtSF"] = (
        X_fe["BsmtFinSF1"].fillna(0) +
        X_fe["BsmtFinSF2"].fillna(0)
    )

    # 4. House age at the moment of sale.
    X_fe["HouseAgeAtSale"] = (
        X_fe["YrSold"] - X_fe["YearBuilt"]
    )

    # 5. Years since last remodeling.
    # Negative values are set to zero because a remodel after the sale
    # is temporally inconsistent with the sale date.
    X_fe["YearsSinceRemodel"] = (
        X_fe["YrSold"] - X_fe["YearRemodAdd"]
    ).clip(lower=0)

    # 6. Total outdoor area.
    outdoor_columns = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch"
    ]

    # min_count preserves NaN if any component is genuinely unknown.
    X_fe["TotalOutdoorArea"] = X_fe[
        outdoor_columns
    ].sum(axis=1, min_count=len(outdoor_columns))

    # 7. Interaction between overall quality and total living area.
    if include_interactions:
        X_fe["QualityTotalAreaInteraction"] = (
            X_fe["OverallQual"] *
            X_fe["TotalAreaIncludingBasement"]
        )

    return X_fe
