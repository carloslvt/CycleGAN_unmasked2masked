import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
AUTOTUNE = tf.data.AUTOTUNE
LAMBDA = 10


def load_image_and_label(file_path):
    """
    Lit et décode l'image
    :param file_path: chemin d'accès aux images
    :return: image + label
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)

    label = 0

    return img, label


def random_crop(image):
    """
    Effectue un recadrage aléatoire sur l'image en utilisant la fonction
    tf.image.random_crop.
    :param image: image en entrée
    :return: image recadrée
    """
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

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


def preprocess_image_train(image, label):
    """
    Applique les fonctions de prétraitement random_jitter et normalize pour l'ensemble d'entraînement.
    :param image: image en entrée
    :param label:
    :return: image prétraitée
    """
    image = random_jitter(image)
    image = normalize(image)

    return image


def preprocess_image_test(image, label):
    """
    Applique la fonction de prétraitement normalize pour l'ensemble de test.
    :param image: image en entrée
    :param label:
    :return: image normalisée
    """
    image = normalize(image)

    return image


def discriminator_loss(real, generated):
    """
    Fonction de perte (BinaryCrossentropy) du discriminateur entre les vrai images et les images générés.
    L'objectif est d'entraîner le discriminateur à distinguer les vraies images des images générées par le générateur.
    :param real: image réelle
    :param generated: image générée
    :return: perte
    """
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    """
    Mesure à quel point les images générées par le générateur sont similaires aux images réelles (fonction de perte).
    L'objectif est d'entraîner le générateur à générer des images de visages masqués réalistes
    à partir des images de visages non masqués et vice versa.
    :param generated: image générée
    :return: perte
    """
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    """
    Calcule la perte de cohérence du cycle entre l'image réelle et l'image cyclée.
    L'image cyclée est l'image obtenue après avoir introduit l'image réelle dans les deux générateurs,
    c'est-à-dire après avoir converti l'image réelle dans le domaine opposé et inversement.
    :param real_image: image réelle
    :param cycled_image: image cyclée
    :return: perte
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    """
    Calcule la perte d'identité entre l'image réelle et la même image. La même image est l'image obtenue en
    introduisant l'image réelle dans le générateur correspondant.
    :param real_image: image réelle
    :param same_image: image identique
    :return: perte
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))

    return LAMBDA * 0.5 * loss


def generate_images(model, test_input):
    """
    Génère une image en utilisant le modèle et le test_input, et affiche les images d'entrée et générées côte à côte
    dans un graphique.
    :param model: generateur
    :param test_input: image test
    :return: plot
    """
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # obtient les valeurs des pixels entre [0, 1] pour les tracer
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(real_x, real_y):
    """
    Fonction d'apprentissage du modèle CycleGAN
    :param real_x: image réelle X (non masqué)
    :param real_y: image réelle Y (masqué)
    :return:
    """
    # persistent est fixé à True car la bande est utilisée plus d'une fois pour calculer les gradients.
    with tf.GradientTape(persistent=True) as tape:
        # generator_g traduit X -> Y
        # generator_f traduit Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x et same_y sont utilisés pour la perte d'identité
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calcul la perte
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calcul les gradients pour le générateur et le discriminateur
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Applique les gradients à l'optimiseur
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))


if __name__ == "__main__":

    # CHARGEMENT DES IMAGES

    train_unmasked_path = "train/unmasked"
    test_unmasked_path = "test/unmasked"
    train_masked_path = "train/masked"
    test_masked_path = "test/masked"

    train_unmasked_files = tf.data.Dataset.list_files(os.path.join(train_unmasked_path, '*.png'))
    test_unmasked_files = tf.data.Dataset.list_files(os.path.join(test_unmasked_path, '*.png'))
    train_masked_files = tf.data.Dataset.list_files(os.path.join(train_masked_path, '*.jpg'))
    test_masked_files = tf.data.Dataset.list_files(os.path.join(test_masked_path, '*.jpg'))

    train_unmasked = train_unmasked_files.map(load_image_and_label).prefetch(AUTOTUNE)
    test_unmasked = test_unmasked_files.map(load_image_and_label).prefetch(AUTOTUNE)
    train_masked = train_masked_files.map(load_image_and_label).prefetch(AUTOTUNE)
    test_masked = test_masked_files.map(load_image_and_label).prefetch(AUTOTUNE)

    # PRETRAITEMENT DES IMAGES

    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    train_unmasked = train_unmasked.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_masked = train_masked.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_unmasked = test_unmasked.cache().map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_masked = test_masked.cache().map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    # AFFICHAGE DES IMAGES

    sample_unmasked = next(iter(train_unmasked))
    sample_masked = next(iter(train_masked))

    plt.subplot(121)
    plt.title('Unmasked Face')
    plt.imshow(sample_unmasked[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Unmasked Face with random jitter')
    plt.imshow(random_jitter(sample_unmasked[0]) * 0.5 + 0.5)

    plt.subplot(121)
    plt.title('Masked Face')
    plt.imshow(sample_masked[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Masked Face with random jitter')
    plt.imshow(random_jitter(sample_masked[0]) * 0.5 + 0.5)

    # IMPORTATION ET REUTILISATION DES MODELES PIX2PIX

    OUTPUT_CHANNELS = 3

    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    to_masked = generator_g(sample_unmasked)
    to_unmasked = generator_f(sample_masked)
    plt.figure(figsize=(8, 8))
    contrast = 8

    # AFFICHAGE DES IMAGES GENEREES SANS APPRENTISSAGE

    imgs = [sample_unmasked, to_masked, sample_masked, to_unmasked]
    title = ['Unmasked Face', 'To Masked Face', 'Masked Face', 'To Unmasked Face']

    for i in range(len(imgs)):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        if i % 2 == 0:
            plt.imshow(imgs[i][0] * 0.5 + 0.5)
        else:
            plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
    plt.show()

    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('Is a real masked face?')
    plt.imshow(discriminator_y(sample_masked)[0, ..., -1], cmap='RdBu_r')

    plt.subplot(122)
    plt.title('Is a real unmasked face?')
    plt.imshow(discriminator_x(sample_unmasked)[0, ..., -1], cmap='RdBu_r')

    plt.show()

    # FONCTION DE PERTE

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # OPTIMISATEUR

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # CHECKPOINTS

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f,
                               discriminator_x=discriminator_x,
                               discriminator_y=discriminator_y,
                               generator_g_optimizer=generator_g_optimizer,
                               generator_f_optimizer=generator_f_optimizer,
                               discriminator_x_optimizer=discriminator_x_optimizer,
                               discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # si un point de contrôle existe, restaure le dernier point de contrôle.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # ENTRAINEMENT

    EPOCHS = 40

    for epoch in tqdm(range(EPOCHS)):
        start = time.time()

        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_unmasked, train_masked)):
            train_step(image_x, image_y)
            if n % 10 == 0:
                print('.', end='')
            n += 1

        clear_output(wait=True)
        # generate_images(generator_g, sample_unmasked)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            generator_g.save(f"models/model_UtoM_{epoch + 1}")
            generator_f.save(f"models/model_MtoU_{epoch + 1}")
            print("Sauvegarde du checkpoint pour l'epoch {} à {}".format(epoch + 1, ckpt_save_path))

        print("Le temps nécessaire pour l'epoch {} est de {}sec\n".format(epoch + 1, time.time() - start))
