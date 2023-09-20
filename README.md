# Real-time-Face-mask-detection
I worked on a real-time face mask detection project using deep learning techniques and achieved an impressive accuracy of 98% on the testing dataset. The objective was to develop a system that can accurately detect whether individuals are wearing masks or not in real-time scenarios. 

To enhance the performance of the model, I augmented the dataset, which involved applying various transformations to the images.

During the training, validation, and testing phases, I utilized a data generator to efficiently load and preprocess the image data. The data generator helped in scaling the pixel values and applying data augmentation techniques such as horizontal flipping, zooming, and shearing. This ensured a robust and diverse dataset for training the model.

For the model architecture, I employed the VGG19 model as a feature extractor. By leveraging the pre-trained weights of the VGG19 model, which was originally trained on the ImageNet dataset, I could effectively extract meaningful features from the face images. The final layers of the model were customized for binary classification, distinguishing between "With Mask" and "Without Mask" classes.

The model achieved an impressive accuracy of 98% on the testing dataset. This high accuracy demonstrates the model's ability to reliably detect whether individuals are wearing masks or not, contributing to the enforcement of mask-wearing protocols and public health measures.

To test the project, you can download the the trained model then run the test script provided in the GitHub repository. The script utilizes real-time video capture and face detection using Haar cascades. It then applies the trained model to classify faces as either wearing masks or not. This real-time face mask detection system provides an efficient way to monitor mask compliance and ensure public safety.

Dataset Link: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

The trained model link: https://www.kaggle.com/datasets/mariamismaiel/maskdetectionmodel



The updated version of the code utilizes two datasets that contain different poses to enhance the accuracy of the classification. This ensures that the model can distinguish between a person wearing a mask and a person covering their mouth with their hands, as it has been trained to recognize different poses. Therefore, if a person puts their hands on their mouth, it will not be considered as wearing a mask.

Dataset
The two datasets used in this project are:

[1] https://www.kaggle.com/datasets/prithwirajmitra/covid-face-mask-detection-dataset/versions/1
[2] https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/versions/1

The trained model link: https://www.kaggle.com/datasets/mariamismaiel/newmodel
