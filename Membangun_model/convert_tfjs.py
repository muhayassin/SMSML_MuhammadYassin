import tensorflow as tf
import tensorflowjs as tfjs

MODEL_DIR = "saved_model"
OUTPUT_DIR = "submission/tfjs_model"

print("[INFO] Loading SavedModel...")
model = tf.keras.models.load_model(MODEL_DIR)

print("[INFO] Converting to TensorFlow.js...")
tfjs.converters.save_keras_model(model, OUTPUT_DIR)

print("[SUCCESS] tfjs_model berhasil dibuat")
