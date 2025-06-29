# Introduction
<p align="justify">
The adoption of AI-based applications is rapidly growing in the modern world across various sectors including intelligent traffic systems and autonomous vehicles. These applications are heavily relied on real-time object detection to function safely and efficiently. I developed a project to identify vehicles by using computer vision techniques from road scene images on the roads of Bangladesh. This project not only focuses on the application of state-of-the-art object detection methods but also contributes toward the development of region-specific autonomous vehicle and traffic monitoring systems.
</p>

# Dataset
<p align="justify">
The dataset used in this project is specifically curated for real-time vehicle detection on Bangladeshi roads. It includes annotated images featuring local road scenarios for region-specific traffic analysis and automation. The dataset is available <a href="https://drive.google.com/drive/u/0/folders/1Oenec-8J2A5QJKMSCWkW_a3keiLYfkkL">here</a>.
</p>

# Objective
<p align="justify">
This project aim to support the development of smart taffic management systems and autonomous driving systems in the heavy congestion traffic environments.
</p>

# Project Woekflow
The code of this project [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L8Fqe_vFLG_jhxtyj5xp6idBLQoGfuG6#scrollTo=0TKnpAAuzMKg)


## Data pre-processing
<p align="justify">
Implementing bounding boxes on the training images to visualize object detection. It helps to check image label correctness and understand the distribution of object classes. 
</p>

## Model Implementation and Evaluation
<p align="justify">
This project focuses on multi-class vehicle classification using YOLOv8 modeltrained on annotated vehicle images. The model is designed to classify images into one of several vehicle categories including cars, buses, trucks, motorbikes, bicycles etc. Several performance metrics were generated :
</p>

-  ### Precision-Recall Curve
<p align="justify">
The thick blue line represents the overall performance across all classes along with a mean avarage precision of 0.422. 'car' and 'three wheelers - CNG' have high AP (Avarage Precision) scores, while 'policecar' and 'scooter' have low AP (Avarage Precision) scores.
</p>

-  ### Precision-Confidence Curve
<p align="justify">
