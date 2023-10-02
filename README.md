# seg_net_from_scratch
In this project, we have diligently constructed SegNet from scratch, and we've taken an additional step by conducting extensive training on both Kaggle and Colab platforms.





<div align="center">
  <h1>YOLOv1: Building Object Detection Model from scratch</h1>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267430814-f1bbe296-e561-45ea-96c3-9afa4c34fdb8.png" width="700" height="300" />
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

### building input-label pair


```
import os
import csv

def get_all_file_names(directory):
    file_names = os.listdir(directory)
    return file_names

def create_dataset(min_idx, max_idx, csv_name, file_names, dist_path):

    with open(f"{dist_path}/{csv_name}.csv", mode="w", newline="") as csv_file:
        for i, fl_name in enumerate(file_names[min_idx:max_idx+1]):
            name = fl_name.split(".")[0]
            img_name = name+".jpg"
            label_name = name+".txt"
            data = [img_name, label_name]
            writer = csv.writer(csv_file)
            writer.writerow(data)


directory_path = "/content/Yolo_Test_Data/annotations"
dist_path="/content/Yolo_Test_Data/"

file_names = get_all_file_names(directory_path)

train_size = [0, int(0.8 * len(file_names))+1]
test_size = [train_size[0], int(0.2 * len(file_names))+1]
val_size = [test_size[0], int(0.2 * len(file_names))+1]


create_dataset(min_idx=train_size[0], max_idx=train_size[1], csv_name="big_train",file_names=file_names, dist_path=dist_path)
create_dataset(min_idx=test_size[0], max_idx=test_size[1], csv_name="big_test",file_names=file_names, dist_path=dist_path)
create_dataset(min_idx=val_size[0], max_idx=val_size[1], csv_name="big_val",file_names=file_names, dist_path=dist_path)

# this is just when building the model
create_dataset(min_idx=0, max_idx=101, csv_name="tiny_train",file_names=file_names, dist_path=dist_path)
create_dataset(min_idx=101, max_idx=121, csv_name="tiny_test",file_names=file_names,dist_path=dist_path)
create_dataset(min_idx=121, max_idx=141, csv_name="tiny_val",file_names=file_names, dist_path=dist_path)
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
