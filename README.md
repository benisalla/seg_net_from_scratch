# seg_net_from_scratch
In this project, we have diligently constructed SegNet from scratch, and we've taken an additional step by conducting extensive training on both Kaggle and Colab platforms.





<div align="center">
  <h1>YOLOv1: Building Object Detection Model from scratch</h1>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272214156-ee453bd8-391a-4f6a-9cf7-71c6dc2e5595.png" width="700" height="300" />
</div>



In this project, we've meticulously developed YOLOv1 from the ground up, and we've gone a step further by conducting comprehensive training on both Kaggle and Colab platforms.


## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#DataPreProcessing)
- [Training](#training)
- [Inference](#inference)
- [About Me](#About)
- [Contact Me](#Contact)
- [License](#license)

## Introduction

The You Only Look Once (YOLO) approach suggests using an end-to-end neural network to make simultaneous predictions of bounding boxes and class probabilities. This contrasts with previous object detection algorithms that adapted classifiers for detection purposes.





## Getting Started

To reuse this implementation, consider the following scenarios:

- If you are utilizing Google Colab, please upload your notebook and proceed to download the Mapillary dataset. Ensure that you save it to your Google Drive and verify that the file paths are correctly configured. Execute the CSV file generation process. It is advisable to follow these steps to ensure accuracy.

- For users on Kaggle, the process is similar to that of Google Colab. However, there is no need to allocate time for dataset construction since Kaggle already provides access to two Mapillary datasets, one of 12GB and another of 29GB.





### Prerequisites

The key requirement here is to utilize a GPU (CUDA). Without it, the process will continue indefinitely, yielding no meaningful results.



## Data Collection

<div align="center">
  <h5>Mapillary Dataset</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108374-d7e965bc-f7e7-4257-8948-ec5a522f5da4.jpeg" width="480" height="220" />
</div>

The Mapillary Vistas dataset is an innovative large-scale street-level image dataset containing 25,000 high-resolution annotated images across 66 object categories with specific instance labels for 37 classes. The annotation is done in a dense and detailed style using polygons to delineate individual objects. Our dataset is five times larger than the total amount of detailed annotations for Cityscapes and includes images from around the world, captured in various weather, seasonal, and hourly conditions. The images come from different imaging devices (mobile phones, tablets, action cameras, professional capture equipment) and photographers with varying levels of experience. This way, our dataset was designed and compiled to cover diversity, rich detail, and geographic extent. As default reference tasks, we define image semantic segmentation and instance-specific image segmentation with the aim of significantly contributing to the development of cutting-edge methods for understanding the visual scene on the road.

<div align="center">
  <h5>Mapillary Dataset Statistics</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108372-63df54d5-af4b-4131-ae1b-456e93223b80.jpeg" width="480" height="220" />
</div>


Here is the structure of this data:

<div align="center">
  <h5>Mapillary Dataset's Structure (Full Dataset in Kaggle - 29GB)</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108366-6fd4d480-10e5-4cb0-a348-154fff461218.jpeg" width="480" height="220" />
</div>

<div align="center">
  <h5>Mapillary Dataset's Structure</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108364-7cb7289e-cdb1-4a82-8b88-dec7bae0c729.jpeg" width="480" height="220" />
</div>





  

## Data PreProcessing

### Pre-Process an image and its mask


```
def process_img_mask(img_path, mask_path):    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (128, 128), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask -= 1
    
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    

    random_flip_layer = tf.keras.layers.RandomFlip(mode="horizontal", seed=random.randint(0, 2000))
    image = random_flip_layer(image)
    mask = random_flip_layer(mask)
        
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
    return (image, mask)

# Map the function to load and preprocess the data
train_ds = tf.data.Dataset.zip((tr_imgs, tr_masks))
train_ds = train_ds.map(lambda im, m: tf.py_function(process_img_mask, [im,m], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.zip((val_imgs, val_masks))
train_ds = val_ds.map(lambda im, m: tf.py_function(process_img_mask, [im,m], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
```








## Training

  loss functions at different times:

<div align="center">
  <h4>Loss Value 1</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108355-906fcb4e-8752-42fe-bef4-aa7424a839db.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Loss Value 2</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108341-7b9afda0-9b79-46b8-bb8c-4e67d889c312.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Loss Value 3</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108377-1b086c6d-76c7-4911-9e88-0029adbd37ac.jpeg" width="700" height="300"/>
</div>





## Inference

<div align="center">
  <h4>example 1</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108361-06a49abc-7ce2-411f-9e9c-6595898323a6.jpeg" width="700" height="300"/>
</div>

<div align="center">
  <h4>example 2</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272108358-c4529364-6be8-4fc0-bcd5-e96a4e72569a.jpeg" width="700" height="300"/>
</div>




## Contact Me
Feel free to reach out to me:
- Email ([ismailbenalla52@gmail.com](mailto:ismailbenalla52@gmail.com))
- linkedin: ([Ben alla Ismail](https://www.linkedin.com/in/ismail-ben-alla-bai/))




## About Me

🎓 I'm Ismail Ben Alla, and I have a deep passion for neural networks 😍. My mission is to assist neural networks in unraveling the mysteries of our universe.</br>
⛵ I'm an enthusiast when it comes to AI, Deep Learning, and Machine Learning algorithms.</br>
✅ I'm an optimist and a dedicated hard worker, constantly striving to push the boundaries of what's possible.</br>
🌱 I'm committed to continuously learning and staying updated with advanced computer science technologies.</br>
😄 I absolutely love what I do, and I'm excited about the endless possibilities in the world of AI and machine learning!</br>

Let's connect and explore the fascinating world of artificial intelligence together! 🤖🌟


<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="ismail-ben-alla-7144b5221/" height="30" width="40" />
  </a>
  <a href="https://instagram.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
</div>





<div align="center">
  <h4>Don't belittle segmentations; they're sharper than you think! 😄</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/272111548-08c84430-36e6-4547-abfa-1127890864c6.gif" width="300" height="300"/>
</div>
