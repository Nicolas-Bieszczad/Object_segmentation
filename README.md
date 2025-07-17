# Object_segmentation
Training an object segmentation model (YOLO11-seg) to segment workers and machinery on working sites

## Introduction

Image segmentation is a fundamental task in computer vision that involves partitioning an image into meaningful regions or segments. These segments correspond to objects, parts of objects, or regions of interest within the scene, enabling more detailed analysis than simple object detection or classification.

Segmentation can be broadly categorized into three types:

- **Semantic Segmentation**: Assigns a class label to every pixel in an image, grouping all pixels of the same category together without distinguishing between different object instances. For example, all workers in an image would be labeled as "worker" without differentiating individual persons.

- **Instance Segmentation**: Not only classifies each pixel but also differentiates between distinct objects of the same class. This means each worker or piece of machinery is individually identified, allowing for object-specific analysis and tracking.

- **Panoptic Segmentation**: Combines the strengths of semantic and instance segmentation by labeling all pixels with both semantic class information and instance identities, covering both "stuff" (amorphous regions like sky or road) and "things" (countable objects like people and machines).

In the context of our worker environment, **instance segmentation** is particularly valuable. Tracking individual workers and machinery allows us to monitor safety compliance effectively—such as verifying if each worker is wearing a hardhat. Assigning unique IDs to workers enables precise tracking and helps reduce redundant alerts—for example, issuing a single alert when a specific worker is found without protective equipment, rather than multiple alerts triggered by the same person across frames. This targeted approach improves both safety management and operational efficiency.

The objective of this project is to train a model to perform instance segmentation on workers and machinery within our worksite environment. The goal is to evaluate segmentation performance and compare it with previous results obtained from object detection tasks. Unlike detection, segmentation provides pixel-level masks for each instance, which is crucial for more precise monitoring and compliance verification.

For this task, we use **YOLOv11-seg**, a recent model from the YOLOv11 family specifically designed for segmentation. YOLOv11-seg combines the real-time speed of YOLO architectures with the capability of generating accurate instance masks. By leveraging this model, we aim to achieve efficient and robust segmentation of workers and equipment to support advanced safety analysis.


## Setup

### Dataset and Preprocessing

The dataset used for training and evaluation was sourced from [Roboflow's "HQ + hole detection, moving objects Computer Vision Project"](https://universe.roboflow.com/thesis-2zk7q/hq-hole-detection-moving-objects). It contains a total of **5,576 images** capturing workers and machinery in various industrial scenarios.

All images were resized to **640 × 640 pixels**, using a _black edge fit_ strategy to preserve aspect ratio and avoid distortion.

To improve model robustness, **data augmentation** was applied:
- Each original image was augmented into two training samples.
- **Horizontal flipping** was used because it does not compromise or distort the segmentation masks, while helping the model generalize better to variations in object orientation.

The dataset was split into three subsets:
- **Training**: 3,903 images  
- **Validation**: 836 images  
- **Testing**: 837 images  

This split follows a 70% / 15% / 15% distribution and ensures balanced representation across all stages of model development and evaluation.



