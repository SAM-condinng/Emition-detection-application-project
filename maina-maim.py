import cv2
import tensorflow as tf

# Load the model
def load_model(path):
    try:
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded and compiled successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

model = load_model('model.hdf5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face cascade for detection
def load_face_cascade(cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error loading face cascade")
        exit()
    return face_cascade

face_cascade = load_face_cascade(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load and process the image
def process_image(image_path, face_cascade, model, emotion_labels):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image")
        print("Image loaded successfully")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected")
        else:
            print(f"Detected {len(faces)} face(s)")

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            input_size = (64, 64)
            face_resized = cv2.resize(face, input_size)
            input_image = face_resized / 255.0
            input_image = tf.expand_dims(input_image, 0)
            input_image = tf.expand_dims(input_image, -1)

            print(f"Input image shape: {input_image.shape}")

            try:
                emotions = model.predict(input_image)[0]
                print(f"Emotion predictions: {emotions}")
                predicted_emotion = emotion_labels[tf.argmax(emotions)]
                print(f"Predicted emotion: {predicted_emotion}")
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

process_image('gachuru2.jpg', face_cascade, model, emotion_labels)
