# Need python3.11

import tensorflow as tf

# 1) Charger ton modèle complet
base_model = tf.keras.models.load_model("BestModelEfficientNetLite.h5")

# 2) Afficher le résumé pour vérifier les noms de couches
base_model.summary()

# 3) Créer un modèle d'inférence sans data augmentation
# On réutilise l'input original
inputs = base_model.input

# On récupère la sortie de la data_augmentation, puis on commence le nouveau modèle APRES cette couche
# Si ta couche s'appelle 'data_augmentation', alors:
x = inputs

# SAUTER la data_augmentation si elle est explicitement une couche dans le modèle
# Exemple : base_model.layers[1] est data_augmentation, base_model.layers[2] est 'efficientnetb0'
for layer in base_model.layers[2:]:
    x = layer(x)

inference_model = tf.keras.Model(inputs=inputs, outputs=x, name="effnetlite_inference")

# 4) Sauvegarder/exporter ce modèle
inference_model.export("saved_model_effnetlite_inference")

# 5) Conversion TFLite avec optimisation
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_effnetlite_inference")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("BestModelEfficientNetLite_inference.tflite", "wb") as f:
    f.write(tflite_model)
