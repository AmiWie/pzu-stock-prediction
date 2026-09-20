import os
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
from datetime import timedelta
from model import StockPriceRNN

def run_prediction():
    # 1. Wczytanie modelu i skalera
    scaler = joblib.load("models/scaler.pkl")
    model = StockPriceRNN()
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval()

    # 2. Pobranie ostatnich 60 dni roboczych, by mieć pewne 30 dni sesyjnych
    df = yf.download("PZU.WA", period="60d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        open_series = df['Open']['PZU.WA'].dropna()
    else:
        open_series = df['Open'].dropna()

    last_30_days = open_series.values[-30:].reshape(-1, 1)
    last_date = open_series.index[-1]

    # 3. Skalowanie i predykcja na kolejną sesję
    scaled_input = scaler.transform(last_30_days)
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(tensor_input).numpy()
    predicted_price = scaler.inverse_transform(pred_scaled).item()

    # Następny dzień sesyjny (jeśli piątek -> poniedziałek)
    next_date = last_date + timedelta(days=1)
    if next_date.weekday() == 5:  # Sobota
        next_date += timedelta(days=2)
    elif next_date.weekday() == 6:  # Niedziela
        next_date += timedelta(days=1)

    print(f"Ostatnia znana sesja: {last_date.strftime('%Y-%m-%d')}, Kurs Otwarcia: {last_30_days[-1][0]:.2f}")
    print(f"Prognoza na kolejną sesję ({next_date.strftime('%Y-%m-%d')}): {predicted_price:.2f} PLN")

    # 4. Zapis / aktualizacja tabeli wyników
    os.makedirs("data", exist_ok=True)
    csv_file = "data/predictions.csv"
    
    if os.path.exists(csv_file):
        pred_df = pd.read_csv(csv_file)
    else:
        # Budujemy historię dla Power BI
        pred_df = pd.DataFrame(columns=["Data", "Cena_Rzeczywista", "Cena_Przewidywana"])

    # Wypełniamy rzeczywistą cenę dla ostatniego znanego dnia
    last_date_str = last_date.strftime('%Y-%m-%d')
    next_date_str = next_date.strftime('%Y-%m-%d')

    # Aktualizacja lub dodanie wierszy
    if last_date_str in pred_df['Data'].values:
        pred_df.loc[pred_df['Data'] == last_date_str, 'Cena_Rzeczywista'] = round(float(last_30_days[-1][0]), 2)
    else:
        new_row = pd.DataFrame([{"Data": last_date_str, "Cena_Rzeczywista": round(float(last_30_days[-1][0]), 2), "Cena_Przewidywana": np.nan}])
        pred_df = pd.concat([pred_df, new_row], ignore_index=True)

    # Wiersz z predykcją na jutro
    if next_date_str in pred_df['Data'].values:
        pred_df.loc[pred_df['Data'] == next_date_str, 'Cena_Przewidywana'] = round(predicted_price, 2)
    else:
        new_row = pd.DataFrame([{"Data": next_date_str, "Cena_Rzeczywista": np.nan, "Cena_Przewidywana": round(predicted_price, 2)}])
        pred_df = pd.concat([pred_df, new_row], ignore_index=True)

    pred_df.drop_duplicates(subset=["Data"], keep="last", inplace=True)
    pred_df.sort_values(by="Data", inplace=True)
    pred_df.to_csv(csv_file, index=False)
    print("Zaktualizowano plik data/predictions.csv!")

if __name__ == "__main__":
    run_prediction()