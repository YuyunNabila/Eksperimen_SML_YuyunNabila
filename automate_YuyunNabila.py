import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DATA_PATH = "data/raw/StudentPerformance.csv"
PROCESSED_DATA_PATH = "data/processed/student_performance_clean.csv"

def load_data(path):
    print("📥 Loading dataset...")
    df = pd.read_csv(path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def handle_missing_values(df):
    print("🧹 Handling missing values...")
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")
    return df

def remove_duplicates(df):
    print("🗑️ Removing duplicate rows...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Rows before: {before}, after: {after}")
    return df


def encode_categorical_features(df):
    print("Encoding categorical features...")
    categorical_cols = df.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    print(f"Encoded columns: {list(categorical_cols)}")
    return df

def handle_outliers(df):
    print("Handling outliers using IQR...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    print("Outlier handling completed")
    return df

def scale_features(df):
    print("📏 Scaling numerical features...")
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Feature scaling applied")
    return df
  
def save_processed_data(df, path):
    print("Saving processed dataset...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved to: {path}")

def main():
    df = load_data(RAW_DATA_PATH)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical_features(df)
    df = handle_outliers(df)
    df = scale_features(df)
    save_processed_data(df, PROCESSED_DATA_PATH)
    print("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()

