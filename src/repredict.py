import os
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
from datetime import timedelta
from ta.momentum import RSIIndicator
from model import StockPriceRNN

def run_prediction():
    # Załadowanie skalerów i modelu
    feature_scaler = joblib.load("models/feature_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")
    
    model = StockPriceRNN(input_size=3, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval()

    # Pobranie danych do wyliczenia RSI 
    df = yf.download("PZU.WA", period="90d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        open_p = df['Open']['PZU.WA']
        close_p = df['Close']['PZU.WA']
        volume = df['Volume']['PZU.WA']
    else:
        open_p = df['Open']
        close_p = df['Close']
        volume = df['Volume']

    data = pd.DataFrame({
        'Open': open_p,
        'Close': close_p,
        'Volume': volume
    }).dropna()

    data['Open_Diff'] = data['Open'].diff()
    data['Vol_Diff'] = np.log1p(data['Volume']) - np.log1p(data['Volume'].shift(1))
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()
    data = data.dropna()

    last_price = float(data['Open'].iloc[-1])
    last_date = data.index[-1]

    # Ostatnie 30 dni sekwencji
    feature_cols = ['Open_Diff', 'Vol_Diff', 'RSI']
    last_30_features = data[feature_cols].iloc[-30:].values

    # Skalowanie i predykcja
    scaled_input = feature_scaler.transform(last_30_features)
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled_diff = model(tensor_input).numpy()

    predicted_diff = target_scaler.inverse_transform(pred_scaled_diff).item()
    predicted_price = round(last_price + predicted_diff, 2)

    # Obliczenie kolejnej daty sesyjnej
    next_date = last_date + timedelta(days=1)
    if next_date.weekday() == 5:
        next_date += timedelta(days=2)
    elif next_date.weekday() == 6:
        next_date += timedelta(days=1)

    print(f"Ostatnia sesja: {last_date.strftime('%Y-%m-%d')} | Kurs Otwarcia: {last_price:.2f} zł")
    print(f"Przewidywana zmiana: {predicted_diff:+.2f} zł | Prognoza na {next_date.strftime('%Y-%m-%d')}: {predicted_price:.2f} zł")

    # Zapis do data/predictions.csv
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