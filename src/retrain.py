import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import yfinance as yf
from ta.momentum import RSIIndicator
from model import StockPriceRNN

def train_model():
    print("Pobieranie pełnej historii PZU")
    df = yf.download("PZU.WA", start="2019-01-01", progress=False)

    # Obsługa MultiIndex z yfinance
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

    # 1. Cecha 1: Zmiana ceny otwarcia 
    data['Open_Diff'] = data['Open'].diff()

    # 2. Cecha 2: Zmiana wolumenu 
    data['Vol_Diff'] = np.log1p(data['Volume']) - np.log1p(data['Volume'].shift(1))

    # 3. Cecha 3: Wskaźnik RSI 
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    # Usunięcie NaN powstałych przez diff i okno RSI
    data = data.dropna().reset_index(drop=True)

    feature_cols = ['Open_Diff', 'Vol_Diff', 'RSI']
    target_col = 'Open_Diff'

    # Podział chronologiczny: 85% train, 15% test
    train_size = int(len(data) * 0.85)
    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]

    # Skalowanie cech wejściowych
    feature_scaler = StandardScaler()
    train_features_scaled = feature_scaler.fit_transform(train_df[feature_cols])
    test_features_scaled = feature_scaler.transform(test_df[feature_cols])

    # Osobny skaler dla targetu (Open_Diff), by łatwo odwrócić wynik
    target_scaler = StandardScaler()
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])

    # Przygotowanie sekwencji 30-dniowych
    n_past = 30
    X_train, y_train = [], []
    for i in range(n_past, len(train_features_scaled)):
        X_train.append(train_features_scaled[i - n_past:i])
        y_train.append(train_target_scaled[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Inicjalizacja modelu z 3 cechami wejściowymi
    model = StockPriceRNN(input_size=3, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"Trening na {len(X_train)} sekwencjach (3 cechy: Price Diff, Volume Diff, RSI)...")
    model.train()
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoka {epoch + 1}/{epochs} - Loss: {loss.item():.5f}")

    # Zapis wag i obu skalerów
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    joblib.dump(feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")
    print("Pomyślnie zaktualizowano model (models/model.pth) oraz skalery!")

if __name__ == "__main__":
    train_model()