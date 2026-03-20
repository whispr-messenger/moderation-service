import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Chargement des classes
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    "../train/dataset/Train",
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical"
)

CLASS_NAMES = list(train_gen.class_indices.keys())

# Quelles classes sont considérées comme junk food
JUNK_CLASSES = {"Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza"}

# Chargement du modèle TFLite
tflite_model_path = "../BestModelEfficientNetLite_inference.tflite"

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded")
print("Input details:", input_details)
print("Output details:", output_details)

# Prétraitement image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {img_path}. Vérifiez que le fichier existe et que le chemin est correct.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # taille attendue par ton modèle

    # Adapter à ce que ton modèle TFLite attend :
    # si le modèle a été converti depuis un modèle qui utilisait rescale=1./255,
    # il vaut mieux normaliser ici :
    img = img.astype("float32")

    img = np.expand_dims(img, axis=0)  # shape (1, 224, 224, 3)
    return img

# Prédiction avec TFLite
def tflite_predict(img_path):
    img = preprocess_image(img_path)

    # Mettre l'input au bon type si nécessaire
    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        # Si ton modèle est quantifié, il peut attendre du uint8.
        # On repasse alors de [0,1] à [0,255] et cast en uint8.
        img_in = (img * 255).astype(np.uint8)
    else:
        img_in = img.astype(input_dtype)

    interpreter.set_tensor(input_index, img_in)
    interpreter.invoke()

    output_index = output_details[0]["index"]
    preds = interpreter.get_tensor(output_index)[0]  # vecteur (7,)
    return preds

# Logique de classification
def predict_image(img_path, threshold=0.9):
    preds = tflite_predict(img_path)      # vecteur de 7 scores / probabilités
    best_idx = np.argmax(preds)           # index de la classe la plus probable
    best_prob = float(preds[best_idx])    # probabilité max
    best_class = CLASS_NAMES[best_idx]    # nom de la classe

    for name, p in zip(CLASS_NAMES, preds):
        print(f"  {name}: {p:.2%}")

    # Est-ce une junk food ?
    is_junk = best_class in JUNK_CLASSES

    if best_prob >= threshold:
        junk_text = "OUI" if is_junk else "NON"
        result = (
            f"Catégorie prédite : {best_class} ({best_prob:.2%})\n"
            f"Junk food : {junk_text}"
        )
    else:
        result = (
            f"Prédiction incertaine (meilleure catégorie : {best_class} "
            f"avec {best_prob:.2%} < {threshold:.0%})\n"
            f"Junk food : Non"
        )

    return result

# Affichage de la prédiction
def show_prediction(img_path, threshold=0.9):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {img_path}. Vérifiez que le fichier existe et que le chemin est correct.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict_image(img_path, threshold=threshold)
    plt.imshow(img)
    plt.axis("off")
    plt.title(result, fontsize=10, color="red")
    plt.show()

# Exemple d'utilisation
img_path = "./images/junk_food.jpg"
show_prediction(img_path, threshold=0.9)
print(predict_image(img_path, threshold=0.9))
