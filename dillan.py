import cv2
import numpy as np
from keras.models import load_model

# Memuat model Anda
model = load_model('model/mobilenet_finetuned_model_multiclass.keras')  # Gantilah dengan path model Anda
face_ref = cv2.CascadeClassifier("utility/haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    # Konversi gambar ke grayscale
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Deteksi wajah
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(50, 50), minNeighbors=5)
    return faces

def drawer_box(frame):
    faces = face_detection(frame)
    for x, y, w, h in faces:
        # Gambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        
        # Potong wajah untuk prediksi
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))  # Resize agar sesuai dengan input model Anda
        face_normalized = face_resized / 255.0  # Normalisasi (jika perlu)
        face_input = np.expand_dims(face_normalized, axis=0)  # Menambahkan dimensi batch
        
        # Prediksi menggunakan model
        predictions = model.predict(face_input)
        label = np.argmax(predictions, axis=1)[0]  # Ambil label dengan probabilitas tertinggi
        confidence = predictions[0][label]  # Ambil tingkat kepercayaan dari label prediksi
        
        # Tentukan nama berdasarkan label (sesuaikan dengan indeks kelas Anda)
        label_names = ["dillan", "marvel"]  # Gantilah dengan nama yang sesuai
        label_name = label_names[label]
        
        # Tampilkan label dan confidence di atas wajah yang terdeteksi
        text = f"{label_name} ({confidence*100:.2f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("Face AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
