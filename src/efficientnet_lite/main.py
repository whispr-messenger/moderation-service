import tensorflow_hub as hub
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

MODEL_URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2"
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"


def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path, image_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def download_labels():
    import requests
    response = requests.get(LABELS_URL)
    labels_path = "imagenet_labels.txt"
    with open(labels_path, "w") as f:
        f.write(response.text)
    return labels_path

def main(image_path):
    # Download labels if not present
    if not os.path.exists("imagenet_labels.txt"):
        labels_path = download_labels()
    else:
        labels_path = "imagenet_labels.txt"
    labels = load_labels(labels_path)

    # Load EfficientNet-Lite model from TF Hub
    model = hub.KerasLayer(MODEL_URL)
    image_size = (224, 224)
    img = preprocess_image(image_path, image_size)
    preds = model(img)
    preds = np.squeeze(preds)
    top_indices = preds.argsort()[-5:][::-1]
    print("Top 5 predictions:")
    for i in top_indices:
        print(f"{labels[i]}: {preds[i]:.4f}")

    # Simple animal keyword list
    animal_keywords = [
        "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "bear", "monkey", "cow", "sheep", "goat", "pig", "rabbit", "deer", "mouse", "rat", "snake", "frog", "elephant", "giraffe", "zebra", "wolf", "fox", "duck", "chicken", "goose", "turkey", "owl", "eagle", "hawk", "falcon", "penguin", "whale", "dolphin", "shark", "crab", "lobster", "bee", "ant", "spider", "butterfly", "bat", "kangaroo", "panda", "camel", "leopard", "cheetah", "hippopotamus", "rhinoceros", "squirrel", "otter", "seal", "moose", "buffalo", "bison", "chimpanzee", "gorilla", "parrot", "peacock", "flamingo", "swan", "crow", "sparrow", "pigeon", "rooster", "hen", "mule", "donkey", "turtoise", "turtle", "lizard", "iguana", "alligator", "crocodile", "octopus", "jellyfish", "starfish", "snail", "slug", "worm", "moth", "locust", "grasshopper", "ladybug", "dragonfly", "firefly", "scorpion", "hedgehog", "porcupine", "armadillo", "platypus", "opossum", "skunk", "badger", "weasel", "ferret", "mongoose", "lemur", "meerkat", "wombat", "wallaby", "koala", "tasmanian devil", "newt", "salamander", "toad", "stingray", "ray", "eel", "guppy", "goldfish", "carp", "trout", "salmon", "bass", "catfish", "pike", "perch", "anchovy", "herring", "mackerel", "tuna", "cod", "halibut", "flounder", "sole", "snapper", "grouper", "barracuda", "marlin", "swordfish", "anglerfish", "seahorse", "sea lion", "walrus", "manatee", "dugong", "narwhal", "orca", "beluga", "seal", "otter", "beaver", "muskrat", "mink", "ermine", "stoat", "shrew", "vole", "lemming", "gerbil", "hamster", "guinea pig", "chinchilla", "capybara", "agouti", "paca", "cavy", "tapir", "antelope", "gazelle", "eland", "kudu", "oryx", "springbok", "impala", "wildebeest", "gemsbok", "hartebeest", "topi", "duiker", "dik-dik", "klipspringer", "steenbok", "sitatunga", "waterbuck", "reedbuck", "kob", "lechwe", "roan", "sable", "addax", "bongo", "nyala", "bushbuck", "mountain goat", "ibex", "chamois", "tahr", "serow", "goral", "muskox", "yak", "saiga", "pronghorn", "vicuna", "guanaco", "alpaca", "llama", "camel", "dromedary", "bactrian camel", "okapi", "giraffe", "elephant", "mammoth", "mastodon", "hyrax", "aardvark", "pangolin", "sloth", "anteater", "armadillo", "raccoon", "coati", "kinkajou", "ringtail", "civet", "genet", "fossa", "margay", "ocelot", "jaguarundi", "puma", "cougar", "mountain lion", "panther", "jaguar", "leopard", "cheetah", "lion", "tiger", "lynx", "bobcat", "caracal", "serval", "snow leopard", "clouded leopard", "sunda clouded leopard", "marbled cat", "flat-headed cat", "fishing cat", "leopard cat", "rusty-spotted cat", "sand cat", "black-footed cat", "jungle cat", "wildcat", "domestic cat", "domestic dog", "wolf", "coyote", "jackal", "dhole", "dingo", "fox", "arctic fox", "red fox", "kit fox", "swift fox", "fennec fox", "bat-eared fox", "raccoon dog", "tanuki", "otter", "badger", "wolverine", "martin", "sable", "mink", "weasel", "ferret", "polecat", "stoat", "ermine", "shrew", "mole", "hedgehog", "tenrec", "solenodon", "desman", "elephant shrew", "aardvark", "pangolin", "sloth", "anteater", "armadillo", "raccoon", "coati", "kinkajou", "ringtail", "civet", "genet", "fossa", "margay", "ocelot", "jaguarundi", "puma", "cougar", "mountain lion", "panther", "jaguar", "leopard", "cheetah", "lion", "tiger", "lynx", "bobcat", "caracal", "serval", "snow leopard", "clouded leopard", "sunda clouded leopard", "marbled cat", "flat-headed cat", "fishing cat", "leopard cat", "rusty-spotted cat", "sand cat", "black-footed cat", "jungle cat", "wildcat", "domestic cat", "domestic dog", "wolf", "coyote", "jackal", "dhole", "dingo", "fox", "arctic fox", "red fox", "kit fox", "swift fox", "fennec fox", "bat-eared fox", "raccoon dog", "tanuki"
    ]
    top_label = labels[top_indices[0]].lower()
    is_animal = any(animal in top_label for animal in animal_keywords)
    print(f"\nClassification: {'animal' if is_animal else 'object'}")

if __name__ == "__main__":
    import sys
    image_path = os.environ.get("IMAGE_PATH")
    if not image_path:
        if len(sys.argv) < 2:
            print("Usage: python main.py")
            sys.exit(1)
        else:
            image_path = sys.argv[1]
    main(image_path)
