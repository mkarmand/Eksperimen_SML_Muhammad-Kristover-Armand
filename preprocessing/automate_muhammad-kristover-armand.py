# automate_preprocessing.py

import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=drop_columns, errors='ignore')
    df = df.dropna().reset_index(drop=True)

    label_cols = ['Sex', 'Embarked']
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df

def split_data(df, random_state=42):
    return train_test_split(df, random_state=random_state)

def save_dataframe(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing dan Split Data")
    parser.add_argument('--input', required=True, help='Path file input (CSV)')
    parser.add_argument('--output-dir', required=True, help='Direktori output (misalnya: preprocessing)')

    args = parser.parse_args()

    df = load_data(args.input)
    df_processed = preprocess_data(df)
    train_df, test_df = split_data(df_processed)

    save_dataframe(train_df, os.path.join(args.output_dir, 'train.csv'))
    save_dataframe(test_df, os.path.join(args.output_dir, 'test.csv'))

    print("✅ Preprocessing dan split selesai. Disimpan di folder:", args.output_dir)
