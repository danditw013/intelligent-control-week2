import cv2
import joblib
import numpy as np 
import pandas as pd

# Muat data warna
color_data = pd.read_csv('colors.csv')

# Ambil fitur warna dan label
X = color_data[['B', 'G', 'R']].values
y = color_data['color_name'].values

# Muat model Decision Tree dan scaler
dt_model = joblib.load('dt_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil dua area di tengah gambar sebagai bounding box
    height, width, _ = frame.shape
    box_size = 50
    center_x, center_y = width // 2, height // 2
    box1 = frame[center_y - box_size:center_y, center_x - box_size:center_x]
    box2 = frame[center_y:center_y + box_size, center_x:center_x + box_size]
    
    # Hitung rata-rata warna dalam bounding box
    avg_color1 = np.mean(box1, axis=(0, 1)).astype(int)
    avg_color2 = np.mean(box2, axis=(0, 1)).astype(int)
    
    # Normalisasi pixel sebelum prediksi
    pixel1_scaled = scaler.transform([avg_color1])
    pixel2_scaled = scaler.transform([avg_color2])
    
    # Prediksi warna utama untuk kedua box
    color_pred1 = dt_model.predict(pixel1_scaled)[0]
    color_pred2 = dt_model.predict(pixel2_scaled)[0]
    
    # Coba dapatkan probabilitas prediksi
    if hasattr(dt_model, "predict_proba"):
        probabilities1 = dt_model.predict_proba(pixel1_scaled)[0]
        probabilities2 = dt_model.predict_proba(pixel2_scaled)[0]
        prob1 = max(probabilities1) * 100  # Probabilitas dalam persen
        prob2 = max(probabilities2) * 100
    else:
        prob1, prob2 = 100.0, 100.0
    
    # Gambar bounding box
    cv2.rectangle(frame, (center_x - box_size, center_y - box_size), (center_x, center_y), (0, 255, 0), 2)
    cv2.rectangle(frame, (center_x, center_y), (center_x + box_size, center_y + box_size), (255, 0, 0), 2)
    
    # Buat bounding box untuk informasi warna di atas frame
    info_x, info_y, info_w, info_h = 30, 10, 400, 70
    cv2.rectangle(frame, (info_x, info_y), (info_x + info_w, info_y + info_h), (255, 255, 255), -1)
    cv2.rectangle(frame, (info_x, info_y), (info_x + info_w, info_y + info_h), (0, 0, 0), 2)
    
    # Tampilkan informasi warna pada frame
    cv2.putText(frame, f'1st: {color_pred1} ({prob1:.2f}%)', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f'2nd: {color_pred2} ({prob2:.2f}%)', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Tampilkan frame
    cv2.imshow('Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
