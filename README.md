# Image Colorization with GAN

## Overview

This project focuses on automatic image colorization using a Generative Adversarial Network (GAN). The model takes grayscale images as input and predicts the corresponding colorized versions by learning the mapping from luminance (L channel) to color information (AB channels) in the LAB color space.

|         Grayscale Input          |         True Color Image          |      Predicted Color Image       |
| :------------------------------: | :-------------------------------: | :------------------------------: |
| ![Grayscale](Ref\Gray_image.png) | ![True Color](Ref\Real_image.png) | ![Predicted](Ref\Pred_image.png) |

## Dataset

The model was trained using the [Image Colorization Dataset](https://www.kaggle.com/datasets/shravankumar9892/image-colorization), which contains grayscale images (L channel) and corresponding color information (AB channels) in `.npy` format. The dataset provides the necessary pairs for training the generator and discriminator of the GAN.

## Data Preparation

To handle the dataset efficiently, a custom `DataGenerator` class was implemented. This class uses **memory mapping** to avoid loading the entire dataset into RAM, which is essential due to its size. The key steps in the data preparation process are:

- **Normalization**: The L channel (grayscale) and AB channels (color) are normalized to suitable ranges before being fed into the model. This ensures that the input data is scaled appropriately for better model performance.
- **Image Filtering**: Not all images in the dataset are suitable for training. The `DataGenerator` filters out images that lack sufficient color variance or contrast. This ensures that the model focuses on learning from high-quality inputs with meaningful color and contrast information.

- **Batch Preparation**: The generator prepares batches of valid grayscale and color image pairs. These batches are then used to train the GAN, ensuring that the model gets a consistent flow of high-quality data.

By optimizing data loading and processing, the `DataGenerator` allows for efficient and scalable training of the model on large datasets like this one.

## Model Architecture

### Generator

The generator is built using a combination of a **ResNet-101 backbone** (pre-trained on ImageNet) and a series of convolutional layers. The ResNet backbone helps the generator learn rich image features, which are crucial for generating high-quality colorized images.

- **Input**: Grayscale image (224x224x1).
- **Output**: Predicted AB channels (224x224x2) for colorization.
- **Skip connections**: Integrated for better feature propagation between layers, enhancing the colorization output.

### Discriminator

The discriminator is designed to classify whether the generated color image is real or fake by comparing the grayscale and color images. This model uses a **PatchGAN** approach, which classifies small patches of the image as real or fake, rather than the entire image. This helps the network focus on local structures, improving the quality of generated images.

![PatchGAN Concept](Ref\Patchgan.png)

### Training Strategy

1. **Pretraining the Generator**: The generator is first pretrained to predict AB channels using mean absolute error (MAE) loss before the adversarial training starts.
2. **GAN Training**: Both the generator and the discriminator are alternately trained. The generator tries to fool the discriminator by producing more realistic color images, while the discriminator learns to distinguish between real and generated images.

- The discriminator is trained on real grayscale-color pairs and fake pairs (from the generator).
- The generator is updated based on how well it fools the discriminator.

## Results Visualization

The model's progress is visually tracked during training by comparing the predicted color images to the true color images.

Below are some examples of colorization results from the training:

![Training Results](Ref\sample.png)

## Future Improvements

1. **Color Representation for Uncommon Colors**:  
   The current model occasionally struggles with accurately predicting uncommon colors (e.g., rare shades or complex hues). Future work could focus on:

   - Training the model on a more diverse dataset with greater variation in colors.
   - Implementing loss functions that emphasize color accuracy, particularly for rare colors.

2. **Progressive GAN Approach**:  
   A **Progressive GAN** architecture could enhance the colorization results by training the model in stages, starting with low-resolution images and gradually increasing the resolution. This approach allows the model to learn global structures first, followed by finer details, leading to more accurate color predictions.
