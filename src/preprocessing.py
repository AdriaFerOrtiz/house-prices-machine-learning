import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


STRUCTURAL_NA_COLUMNS = [
    "Alley",
    "PoolQC",
    "MiscFeature",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
]


def apply_semantic_rules(df):
    """Aplica reglas deterministas basadas en el significado
    de las variables del dataset.
    """
    df = df.copy()

    # NA significa que estas características no existen.
    for col in STRUCTURAL_NA_COLUMNS:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(df[col].notna(), "None")

    # Separar ausencia de mampostería y tipo desconocido.
    df["MasVnrType"] = df["MasVnrType"].astype("object")

    type_missing = df["MasVnrType"].isna()
    no_veneer = type_missing & df["MasVnrArea"].eq(0)
    unknown_type = type_missing & ~df["MasVnrArea"].eq(0)

    df.loc[no_veneer, "MasVnrType"] = "None"
    df.loc[unknown_type, "MasVnrType"] = "Unknown"

    # Si no existe sótano, las cantidades ausentes equivalen a cero.
    basement_numeric = [
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
    ]

    no_basement = df["BsmtQual"].eq("None")

    for col in basement_numeric:
        df.loc[no_basement & df[col].isna(), col] = 0

    # Si no existe garaje, imputar como cero solo cantidades
    # ausentes de capacidad y superficie. No el año de construcción.
    no_garage = df["GarageType"].eq("None")

    for col in ["GarageCars", "GarageArea"]:
        df.loc[no_garage & df[col].isna(), col] = 0

    # Estas columnas representan categorías, no cantidades continuas.
    for col in ["MSSubClass", "MoSold"]:
        df[col] = df[col].astype("string")

    return df


def build_preprocessor(X_reference, scale_numeric=False):
    """Construye un preprocesador para las columnas de entrada."""
    reference_clean = apply_semantic_rules(X_reference)

    numeric_features = (
        reference_clean
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = [
        col for col in reference_clean.columns
        if col not in numeric_features
    ]

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        # El dataset es pequeño; usamos matriz densa para admitir
        # también los estimadores que no aceptan matrices dispersas.
        sparse_threshold=0.0,
    )


def build_pipeline(X_reference, estimator, scale_numeric=False):
    """Encadena reglas semánticas, preprocessing y modelo."""
    return Pipeline(
        steps=[
            (
                "semantic_rules",
                FunctionTransformer(
                    apply_semantic_rules,
                    validate=False,
                ),
            ),
            (
                "preprocessor",
                build_preprocessor(
                    X_reference,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", estimator),
        ]
    )
