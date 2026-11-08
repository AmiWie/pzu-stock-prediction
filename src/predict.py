# src/predict.py
import os
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
from datetime import timedelta
from model import StockPriceRNN

def run_prediction():
    scaler = joblib.load("models/scaler.pkl")
    model = StockPriceRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval()

    # Pobieramy ostatnie 60 dni roboczych
    df = yf.download("PZU.WA", period="60d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        open_series = df['Open']['PZU.WA'].dropna()
    else:
        open_series = df['Open'].dropna()

    prices = open_series.values
    last_price = float(prices[-1])
    last_date = open_series.index[-1]

    # Obliczamy ostatnie 30 dziennych zmian
    diffs = np.diff(prices)[-30:].reshape(-1, 1)

    scaled_diffs = scaler.transform(diffs)
    tensor_input = torch.tensor(scaled_diffs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled_diff = model(tensor_input).numpy()
    
    # Przewidziana zmiana ceny w PLN 
    predicted_diff = scaler.inverse_transform(pred_scaled_diff).item()
    
    # Nowa cena = ostatnia znana cena + przewidziana zmiana
    predicted_price = round(last_price + predicted_diff, 2)

    # Następny dzień sesyjny
    next_date = last_date + timedelta(days=1)
    if next_date.weekday() == 5:
        next_date += timedelta(days=2)
    elif next_date.weekday() == 6:
        next_date += timedelta(days=1)

    print(f"Ostatnia sesja: {last_date.strftime('%Y-%m-%d')} | Kurs: {last_price:.2f} zł")
    print(f"Przewidywana zmiana: {predicted_diff:+.2f} zł | Prognoza na {next_date.strftime('%Y-%m-%d')}: {predicted_price:.2f} zł")

    # Zapis do CSV
    os.makedirs("data", exist_ok=True)
    csv_file = "data/predictions.csv"
    
    if os.path.exists(csv_file):
        pred_df = pd.read_csv(csv_file)
    else:
        pred_df = pd.DataFrame(columns=["Data", "Cena_Rzeczywista", "Cena_Przewidywana"])

    last_date_str = last_date.strftime('%Y-%m-%d')
    next_date_str = next_date.strftime('%Y-%m-%d')

    if last_date_str in pred_df['Data'].values:
        pred_df.loc[pred_df['Data'] == last_date_str, 'Cena_Rzeczywista'] = round(last_price, 2)
    else:
        new_row = pd.DataFrame([{"Data": last_date_str, "Cena_Rzeczywista": round(last_price, 2), "Cena_Przewidywana": np.nan}])
        pred_df = pd.concat([pred_df, new_row], ignore_index=True)

    if next_date_str in pred_df['Data'].values:
        pred_df.loc[pred_df['Data'] == next_date_str, 'Cena_Przewidywana'] = predicted_price
    else:
        new_row = pd.DataFrame([{"Data": next_date_str, "Cena_Rzeczywista": np.nan, "Cena_Przewidywana": predicted_price}])
        pred_df = pd.concat([pred_df, new_row], ignore_index=True)

    pred_df.drop_duplicates(subset=["Data"], keep="last", inplace=True)
    pred_df.sort_values(by="Data", inplace=True)
    pred_df.to_csv(csv_file, index=False)
    print("Zaktualizowano plik data/predictions.csv!")

if __name__ == "__main__":
    run_prediction()