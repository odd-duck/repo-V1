# Gerekli kütüphaneleri içe aktar
import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import pytz
import ta
import os
import time
import threading
import streamlit as st

# === AYARLAR ===
coin_symbol = 'ETH/USDT'
time_frame = '1h'
data_length = 1000
sequence_length = 50
local_timezone = 'Europe/Istanbul'

# Veriyi çek
print("Veriler çekiliyor...")
exchange = ccxt.gateio()
exchange.rateLimit = 1000

# HTML ve CSS dosyalarını oku
def load_html_css():
    with open("index.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
    with open("styles.css", "r", encoding="utf-8") as css_file:
        css_content = css_file.read()
    return html_content, css_content

# Streamlit arayüzü ve canlı grafik için fonksiyon
def live_chart(exchange, coin_symbol, time_frame, local_timezone):
    placeholder = st.empty()  # Grafiği güncellemek için bir placeholder oluştur
    while True:  # Sürekli güncelleme için döngü
        # Veriyi çek
        since = exchange.milliseconds() - 100 * 3600 * 1000  # Son 100 veri noktası
        ohlcv = exchange.fetch_ohlcv(coin_symbol, time_frame, since)
        df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], unit='ms', utc=True)
        df_live['timestamp'] = df_live['timestamp'].dt.tz_convert(local_timezone)

        # Grafik oluştur
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_live['timestamp'],
            open=df_live['open'],
            high=df_live['high'],
            low=df_live['low'],
            close=df_live['close'],
            name="Gerçek Fiyat"
        ))
        fig.update_layout(
            title=f'{coin_symbol} Canlı Fiyat Grafiği ({time_frame})',
            xaxis_title='Zaman',
            yaxis_title='Fiyat',
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        # Grafiği placeholder'a yerleştir
        with placeholder.container():
            st.plotly_chart(fig, use_container_width=True)

        # 10 saniye bekle ve grafiği güncelle
        time.sleep(10)

# Streamlit arayüzü
st.set_page_config(layout="wide", page_title="Kripto Tahmin Uygulaması", page_icon="📈")

# HTML ve CSS'yi yükle
html_content, css_content = load_html_css()

# CSS'yi HTML'e göm
html_with_css = html_content.replace(
    '<link rel="stylesheet" href="styles.css">',
    f'<style>{css_content}</style>'
)

# HTML'i Streamlit'e göm
st.markdown(html_with_css, unsafe_allow_html=True)

# Canlı grafiği chart-placeholder'a yerleştir
with st.container():
    st.markdown('<div id="chart-placeholder">', unsafe_allow_html=True)
    threading.Thread(target=live_chart, args=(exchange, coin_symbol, time_frame, local_timezone), daemon=True).start()
    st.markdown('</div>', unsafe_allow_html=True)

# Tahmin parametrelerini form-placeholder'a yerleştir
with st.container():
    st.markdown('<div id="form-placeholder">', unsafe_allow_html=True)
    coin_symbol = st.text_input("Coin Sembolü (örneğin ETH/USDT):", value=coin_symbol)
    time_frame = st.selectbox("Zaman Dilimi:", options=['1h', '4h', '12h', '1d'], index=0)
    data_length = st.slider("Veri Uzunluğu (kaç veri noktası):", min_value=100, max_value=3000, value=data_length)
    sequence_length = st.slider("Sekans Uzunluğu:", min_value=10, max_value=100, value=sequence_length)
    local_timezone = st.selectbox("Saat Dilimi:", options=['Europe/Istanbul', 'America/New_York', 'Asia/Tokyo'], index=0)

    # Tahmin Et butonu
    if st.button("Tahmin Et"):
        with st.spinner("Tahmin yapılıyor..."):
            # Veriyi çek
            since = exchange.milliseconds() - data_length * 3600 * 1000
            ohlcv = exchange.fetch_ohlcv(coin_symbol, time_frame, since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(local_timezone)

            # Teknik göstergeler
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df = df.bfill()

            # Sentiment simülasyonu
            np.random.seed(42)
            df['sentiment'] = np.random.uniform(-1, 1, size=len(df))

            # Veri ön işleme
            data = df[['close', 'volume', 'rsi', 'macd', 'macd_signal', 'sentiment']]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # Sekans oluşturma
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)

            # Eğitim/test ayrımı
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

            # Tahmin
            predicted_prices = model.predict(X_test, verbose=0)
            predicted_prices = scaler.inverse_transform(
                np.concatenate((predicted_prices, np.zeros((len(predicted_prices), data.shape[1]-1))), axis=1)
            )[:, 0]
            real_prices = scaler.inverse_transform(
                np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), data.shape[1]-1))), axis=1)
            )[:, 0]

            # Metrikler
            mae = mean_absolute_error(real_prices, predicted_prices)
            mse = mean_squared_error(real_prices, predicted_prices)
            rmse = np.sqrt(mse)
            r2 = r2_score(real_prices, predicted_prices)

            # Bir sonraki fiyat tahmini
            last_sequence = scaled_data[-sequence_length:]
            last_sequence = np.reshape(last_sequence, (1, sequence_length, last_sequence.shape[1]))
            next_price_scaled = model.predict(last_sequence, verbose=0)
            next_price = scaler.inverse_transform(
                np.concatenate((next_price_scaled, np.zeros((1, data.shape[1]-1))), axis=1)
            )[:, 0]
            next_timestamp = df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)

            # Sonuçları results-placeholder'a yerleştir
            with st.container():
                st.markdown('<div id="results-placeholder">', unsafe_allow_html=True)
                st.markdown("--- MODEL BAŞARI METRİKLERİ ---")
                st.write(f"MAE: {mae:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"R^2 Skoru: {r2:.4f}")
                st.write(f"Tahmin edilen bir sonraki fiyat: {next_price[0]:.4f} ({next_timestamp})")
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
