import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Загрузить обученную модель
model = load_model('fdc.h5')


#cap = cv2.VideoCapture('video.mp4')  # Заменить 'video.mp4' на путь к вашему видеофайлу
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #обработка кадра
    frame_resized = cv2.resize(frame, (256, 256))  # Масштабирование до размера модели
    frame_normalized = frame_resized / 255.0  # Нормализация

    # Предсказание
    predictions = model.predict(np.expand_dims(frame_normalized, axis=0))

    if predictions[0][0] > 0.5:
        # Если модель обнаружила лицо, используйте детектор лиц OpenCV для получения координат лица
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Отрисовка ректангла
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # отображение
    cv2.imshow('DeepEYE', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
