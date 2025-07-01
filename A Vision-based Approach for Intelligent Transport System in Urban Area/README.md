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
This plot helps to choose an optimal confidence threshold associated with the balanced precision and recall to make the model more selective.
</p>

-  ### F1-Confidence Curve
<p align="justify">
The overall best F1 is approximately 0.38 at a threshold of .357. 'Car' and 'Bus' classes achieve higher F1 scores along with the optimal confidence threshold.
</p>

-  ### Recall-Confidence Curve
<p align="justify">
This curve shows how recall differs by adjusting the confidence threshold. Higher recall is seen at low confidence threshold (e.g. car, bus).
</p>

-  ### Confusion Matrix
<p align="justify">
This model shows strong performance in high sample classes along with some confusion between similar categories (e.g. minivan, van).
</p>

-  ### Training Metrics and Loss Visualization
<p align="justify">
A grid of line plots to visualize track model convergence by observing loss reduction over epochs.
</p>

-  ### Evaluation & object detection
<p align="justify">
The trained object detection YOLOv8 model demonstrates evaluation on the validation image set by specifying the trained model checkpoint and the validation image directory. Each selected image is displayed with the detected objects by accessing the directory containing the predicted results and randomly selects 30 images for visualization. 
</p>

## Conclusion
<p align="justify">
In this project, an object detection model shows promising performance on a diverse dataset to detect and classify multiple object categories accurately. The model is capable of identifying most target objects with reasonable precision and recall. Though the evaluation results on both quantitive metrics and qualitative visualizations indicate that the model performs well on several classes, there are remain opportunities for improvement. 
</p>

## Future Work
<p align="justify">

-  Explore real-time deployment options and optimize the model for mobile platforms.
-  Integrating post-processing steps such as Non-Maximum Suppression (NMS) optimization for better handling or overlapping detections.
-  Implement transfer learning approaches and ensemble methods to further boost performance.

</p>
