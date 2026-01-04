from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt


datagen = ImageDataGenerator(rescale=1./255)

CLASS_NAMES = None
JUNK_CLASSES = {"Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza"}
model = None


def load_resources():
    """Charge train_gen, CLASS_NAMES et le modèle uniquement quand on en a besoin."""
    global CLASS_NAMES, model

    if CLASS_NAMES is None:
        train_gen = datagen.flow_from_directory(
            "./train/dataset/Train",
            target_size=(224, 224),
            batch_size=8,
            class_mode="categorical"
        )
        CLASS_NAMES = list(train_gen.class_indices.keys())

    if model is None:
        model = load_model("./BestModelEfficientNetLite.h5")
        print("model loaded")


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {img_path}. Vérifiez que le fichier existe et que le chemin est correct.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # taille attendue par ton modèle
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path, threshold=0.9):
    load_resources()
    img = preprocess_image(img_path)
    preds = model.predict(img)[0]          # vecteur de 7 probabilités
    best_idx = np.argmax(preds)            # index de la classe la plus probable
    best_prob = float(preds[best_idx])     # probabilité max
    best_class = CLASS_NAMES[best_idx]     # nom de la classe
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


if __name__ == "__main__":
    # Exemple d'utilisation
    img_path = "./test_images/fries.jpg"
    show_prediction(img_path, threshold=0.9)
    print(predict_image(img_path, threshold=0.9))


