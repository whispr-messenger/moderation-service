from pathlib import Path
import imghdr
import os

# Dossier racine du dataset
# Exemple WSL2 : "/workspace/src/efficientnet_lite_gpu/train/dataset"
# Exemple Windows : r"C:\Users\Nathan\Desktop\moderation-service\src\efficientnet_lite_gpu\train\dataset"
ROOT_DIR = "./src/efficientnet_lite_gpu/train/dataset"

# Extensions que tu considères comme images dans ton dataset
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Types réellement acceptés par TensorFlow pour tf.image.decode_image
TF_ACCEPTED_TYPES = {"bmp", "gif", "jpeg", "png"}

bad_files = []

for filepath in Path(ROOT_DIR).rglob("*"):
    if not filepath.is_file():
        continue

    ext = filepath.suffix.lower()
    if ext not in IMAGE_EXTS:
        # Si tu veux aussi lister/supprimer les fichiers non image dans ces dossiers, enlève ce "continue"
        continue

    # img_type = type réel détecté à partir du contenu du fichier
    img_type = imghdr.what(filepath)

    if img_type is None:
        print(f"PAS une image valide (contenu): {filepath}")
        bad_files.append(filepath)
    elif img_type not in TF_ACCEPTED_TYPES:
        print(f"Type NON supporté par TF ({img_type}): {filepath}")
        bad_files.append(filepath)

print()
print("Nombre de fichiers problématiques détectés (côté TensorFlow):", len(bad_files))

# Si tu veux les supprimer automatiquement, décommente la boucle suivante:
# for f in bad_files:
#     try:
#         os.remove(f)
#         print("Supprimé:", f)
#     except Exception as e:
#         print("Erreur en supprimant", f, "->", e)
