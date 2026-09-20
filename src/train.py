import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
from model import StockPriceRNN

#  Pobranie danych z GPW 
print("Pobieranie danych PZU")
df = yf.download("PZU.WA", start="2018-01-01", progress=False)

# Jeśli yfinance zwróci MultiIndex, wybieramy kolumnę Open
if isinstance(df.columns, pd.MultiIndex):
    prices = df['Open']['PZU.WA'].dropna().values.reshape(-1, 1)
else:
    prices = df['Open'].dropna().values.reshape(-1, 1)

# Podział danych (80% train, 20% test)
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]
test_prices = prices[train_size:]

# Skalowanie (fit tylko na zbiorze treningowym!)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_prices)
test_scaled = scaler.transform(test_prices)

full_scaled = np.vstack([train_scaled, test_scaled])

# Tworzenie sekwencji 30-dniowych
n_past = 30
X, y = [], []
for i in range(n_past, len(full_scaled)):
    X.append(full_scaled[i - n_past:i])
    y.append(full_scaled[i])

X, y = np.array(X), np.array(y)

split_idx = train_size - n_past
X_train, y_train = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(y[:split_idx], dtype=torch.float32)
X_test, y_test = torch.tensor(X[split_idx:], dtype=torch.float32), torch.tensor(y[split_idx:], dtype=torch.float32)

# Model i Trening
model = StockPriceRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Rozpoczynanie treningu")
num_epochs = 300  
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoka {epoch + 1}/{num_epochs}, Loss: {loss.item():.5f}")

# Zapis modelu i skalera
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
joblib.dump(scaler, "models/scaler.pkl")
print("Zapisano model (models/model.pth) i skaler (models/scaler.pkl)!")