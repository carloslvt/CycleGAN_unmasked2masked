# Génération de visages masqués réalistes à l'aide de GANs

## Introduction

La reconnaissance et l'identification faciales sont des aspects essentiels de la technologie moderne et sont entravées par l'avènement des masques faciaux pour lutter contre le Covid-19. De vastes ensembles de données de visages masqués sont nécessaires pour améliorer l'identification faciale avec des masques. De tels ensembles de données peuvent être produits à l'aide de GAN. Dans ce travail, des images réalistes de visages masqués sont générées à l'aide d'un modèle CycleGAN. 

## Dataset 

Un ensemble de données de 133 783 visages masqués (MaskedFace-Net), formaté à son tour pour fonctionner avec un ensemble de données de 70 000 visages (FFHQ), a été utilisé. L'ensemble de données de visages masqués contient des masques artificiels superposés aux visages, au lieu de véritables images de masques. Afin de transférer les fichiers et d'accélérer l'entrainement, un sous-ensemble de 2 000 images d'apprentissage et de 1 000 images de test a été utilisé à partir de chacun de ces grands ensembles de données.

## CycleGAN

Des modèles CycleGAN ont été utilisés pour des tâches telles que la conversion d'images de chevaux en zèbres, avec de meilleures performances que des modèles GAN de structure similaire. Une approche CycleGAN a donc été utilisée pour ajouter des masques à des images de visages. La perte CycleGAN consiste principalement en une perte GAN avec deux discriminateurs entraînés Dx et Dy et des générateurs G et F.

Le modèle CycleGAN a été entraîné pour ajouter des masques aux visages non masqués. Les réseaux générateurs ont été formés à partir de trois convolutions initiales, de neuf blocs UNET convolutifs à 64 canaux, de deux convolutions fractionnées et d'une convolution finale pour réduire la sortie à trois canaux. Le discriminateur CNN est évalué sur des parcelles d'images 70x70 qui se chevauchent. Trois couches de convolution avec un pas de deux sont appliquées à chaque image pour augmenter la profondeur d'activation et réduire la largeur d'activation. La couche intermédiaire remodelée est atténuée et introduite dans une sortie sigmoïde à un seul neurone entièrement connecté.
