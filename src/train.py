# src/train.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import yfinance as yf
from model import StockPriceRNN

print("Pobieranie danych PZU")
df = yf.download("PZU.WA", start="2019-01-01", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    prices = df['Open']['PZU.WA'].dropna().values
else:
    prices = df['Open'].dropna().values

# Obliczamy zmiany dzienne (cenę dzisiejszą minus wczorajszą)
# Wartości będą oscylować wokół zera (np. -1.2, +0.5, +1.8 zł)
price_diffs = np.diff(prices).reshape(-1, 1)

# Używamy StandardScaler 
scaler = StandardScaler()
diffs_scaled = scaler.fit_transform(price_diffs)

# 30 ostatnich zmian przewiduje 31. zmianę
n_past = 30
X, y = [], []
for i in range(n_past, len(diffs_scaled)):
    X.append(diffs_scaled[i - n_past:i])
    y.append(diffs_scaled[i])

X, y = np.array(X), np.array(y)

# Podział chronologiczny (85% train, 15% test)
train_size = int(len(X) * 0.85)
X_train, y_train = torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(y[:train_size], dtype=torch.float32)
X_test, y_test = torch.tensor(X[train_size:], dtype=torch.float32), torch.tensor(y[train_size:], dtype=torch.float32)

model = StockPriceRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)
criterion = nn.HuberLoss()  # HuberLoss jest odporniejszy na nagłe skoki kursu niż MSE
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

print("Trenowanie modelu na zmianach cen")
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 40 == 0:
        print(f"Epoka {epoch + 1}/{num_epochs}, Loss: {loss.item():.5f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
joblib.dump(scaler, "models/scaler.pkl")
print("Zapisano zaktualizowany model i skaler!")