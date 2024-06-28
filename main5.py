import cv2
import tensorflow as tf

# Load the pre-trained model for emotion detection
try:
    model = tf.keras.models.load_model('model.hdf5', compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded and compiled successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the face detection cascade
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Error loading face cascade")
    exit()

# Load and process the image
image_path = 'gachuru2.jpg'
try:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read image")
    print("Image loaded successfully")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected")
    else:
        print(f"Detected {len(faces)} face(s)")

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y + h, x:x + w]

        # Resize the face image to match the input size expected by the model
        input_size = (64, 64)  # Updated to 64x64
        face_resized = cv2.resize(face, input_size)

        # Normalize the face image
        input_image = face_resized / 255.0

        # Reshape the image to match the input shape expected by the model
        input_image = tf.expand_dims(input_image, 0)
        input_image = tf.expand_dims(input_image, -1)

        # Print the shape of the input image for debugging
        print(f"Input image shape: {input_image.shape}")

        # Perform emotion detection
        try:
            emotions = model.predict(input_image)[0]
            print(f"Emotion predictions: {emotions}")
            predicted_emotion = emotion_labels[tf.argmax(emotions)]
            print(f"Predicted emotion: {predicted_emotion}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted emotion on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, predicted_emotion, (x, y - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with emotion predictions
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")