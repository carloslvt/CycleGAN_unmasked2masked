import streamlit as st
import os
import matplotlib.pyplot as plt
from cyclegan import load_image_and_label
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1

def generate_images(model, test_input):
    """
    Génère une image en utilisant le modèle et le test_input, et affiche les images d'entrée et générées côte à côte
    dans un graphique.
    :param model: generateur
    :param test_input: image test
    :return: plot
    """
    prediction = model(test_input)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 12))
    axs[0].imshow(test_input[0] * 0.5 + 0.5)
    axs[0].set_title("Input Image")
    axs[1].imshow(prediction[0] * 0.5 + 0.5)
    axs[1].set_title("Predicted Image")
    for ax in axs:
        ax.axis("off")

    return fig


def random_crop(image):
    """
    Effectue un recadrage aléatoire sur l'image en utilisant la fonction
    tf.image.random_crop.
    :param image: image en entrée
    :return: image recadrée
    """
    cropped_image = tf.image.random_crop(
        image, size=[256, 256, 3])

    return cropped_image


def normalize(image):
    """
    Normalise l'image en convertissant ses valeurs en float, en la mettant à l'échelle dans la plage [-1, 1].
    :param image: image en entrée
    :return: image normalisée
    """
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image


def random_jitter(image):
    """
    Applique une augmentation de données en redimensionnant l'image à 286x286 pixels, en effectuant un recadrage
    aléatoire pour obtenir une image de 256x256 pixels, et en effectuant une réflexion horizontale aléatoire.
    :param image: image en entrée
    :return: image modifiée
    """
    # redimensionne en 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # effecture un crop aléatoire 256 x 256 x 3
    image = random_crop(image)
    # effectue une réflexion horizontale aléatoire
    image = tf.image.random_flip_left_right(image)

    return image

def preprocess_image(image, label):
    """
    Applique les fonctions de prétraitement random_jitter et normalize pour l'ensemble d'entraînement.
    :param image: image en entrée
    :param label:
    :return: image prétraitée
    """
    image = random_jitter(image)
    image = normalize(image)

    return image



st.title("Importer votre image")
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
unmasked_path = "C:/Users/louva/OneDrive/Documents/Cours/M2 SDTS/Deep_learning/PROJET_GAN/unmasked/test"
masked_path = "C:/Users/louva/OneDrive/Documents/Cours/M2 SDTS/Deep_learning/PROJET_GAN/masked/test"

if uploaded_file is not None:

    option = st.selectbox(
        'Quelle modification voulez-vous effectuer',
        ('Démasqué vers Masqué', 'Masqué vers Démasqué'))

    if option == 'Démasqué vers Masqué':
        model = tf.keras.models.load_model("models/model_UtoM_30")

        filename = uploaded_file.name
        file_path = os.path.join(unmasked_path, filename)
        file_path = file_path.replace("\\", "/")

        test_files = tf.data.Dataset.list_files(file_path)
        test = test_files.map(load_image_and_label).prefetch(AUTOTUNE)
        new_test = test.cache().map(
            preprocess_image, num_parallel_calls=AUTOTUNE).cache().shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        sample = next(iter(new_test))
        st.pyplot(generate_images(model, sample))

    else :
        model = tf.keras.models.load_model("models/model_MtoU_30")

        filename = uploaded_file.name
        file_path = os.path.join(masked_path, filename)
        file_path = file_path.replace("\\", "/")

        test_files = tf.data.Dataset.list_files(file_path)
        test = test_files.map(load_image_and_label).prefetch(AUTOTUNE)
        new_test = test.cache().map(
            preprocess_image, num_parallel_calls=AUTOTUNE).cache().shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE)

        sample = next(iter(new_test))
        st.pyplot(generate_images(model, sample))

else:
    st.write("Veuillez télécharger une image")
