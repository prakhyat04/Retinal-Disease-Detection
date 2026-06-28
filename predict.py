import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("retinal_model.h5")

test_folder = "test_images"

print("Starting prediction...\n")

# ======================
# LOOP IMAGES
# ======================
for img_name in os.listdir(test_folder):

    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):

        img_path = os.path.join(test_folder, img_name)

        try:
            # ======================
            # LOAD + PREPROCESS
            # ======================
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ======================
            # PREDICT
            # ======================
            score = float(model.predict(img_array, verbose=0)[0][0])

            # ======================
            # LOGICAL INTERPRETATION
            # ======================
            # Model logic:
            # 0 → cataract
            # 1 → normal

            if score <= 0.35:
                result = "cataract"
                confidence = (1 - score) * 100
                level = "HIGH CONFIDENCE"

            elif score >= 0.65:
                result = "normal"
                confidence = score * 100
                level = "HIGH CONFIDENCE"

            else:
                result = "uncertain"
                confidence = (0.5 - abs(0.5 - score)) * 200  # mid confidence logic
                level = "LOW CONFIDENCE"

            # ======================
            # OUTPUT
            # ======================
            print("=================================")
            print("Image:", img_name)
            print("Raw Score:", round(score, 4))
            print("Prediction:", result)
            print("Confidence:", f"{confidence:.2f}%")
            print("Level:", level)
            print("=================================\n")

        except Exception as e:
            print("Error:", img_name, e)

print("Done.")