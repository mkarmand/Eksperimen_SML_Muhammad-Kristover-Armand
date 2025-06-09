# automate_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(filepath):
    """Membaca file CSV menjadi DataFrame."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocessing data Titanic:
    - Drop kolom tidak penting
    - Hapus baris dengan missing values
    - Encode fitur kategorikal ('Sex', 'Embarked')
    - Scaling fitur numerik ('Age', 'Fare')
    """
    # Drop kolom tidak penting
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    # Hapus baris dengan missing value
    df = df.dropna().reset_index(drop=True)
    
    # Encode fitur kategorikal
    label_cols = ['Sex', 'Embarked']
    le_dict = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # simpan encoder jika diperlukan
    
    # Scaling fitur numerik
    scaler = MinMaxScaler()
    scale_cols = ['Age', 'Fare']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    return df, le_dict, scaler

def save_preprocessed(df, filepath):
    """Simpan DataFrame hasil preprocessing ke file CSV."""
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    # Input dan output path
    input_file = "titanic_raw/data.csv"
    output_file = "preprocessing/titanic_preprocessing.csv"
    
    df = load_data(input_file)
    df_preprocessed, encoders, scaler = preprocess_data(df)
    save_preprocessed(df_preprocessed, output_file)
    print(f"Preprocessing selesai, data tersimpan di {output_file}")
