import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.lite.Interpreter(model_path='mobilenetv2_waste.tflite')
model.allocate_tensors()

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Test dengan gambar
img = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Inference
input_details = model.get_input_details()
output_details = model.get_output_details()

model.set_tensor(input_details[0]['index'], img_array)
model.invoke()

output = model.get_tensor(output_details[0]['index'])
pred_class = np.argmax(output[0])
confidence = output[0][pred_class]

print(f"Prediksi: {labels[pred_class]}")
print(f"Confidence: {confidence:.2%}")