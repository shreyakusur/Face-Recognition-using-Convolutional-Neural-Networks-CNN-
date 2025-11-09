This project builds and analyzes a Convolutional Neural Network (CNN) for face recognition.
The main goal is to compare how input image size and Dropout (0.5) affect model accuracy and overfitting.


Project Overview:
This experiment tests CNN performance under different configurations:
Image sizes: 32×32, 64×64, and 128×128
With and without Dropout (0.5)
3 convolutional layers to extract hierarchical facial features

How It Works:
Load Dataset — Images are rescaled and split into training/testing sets.
Build CNN — The network is created with adjustable input size and dropout rate.
Train Model — Each version (size + dropout setting) is trained for 10 epochs.
Compare Results — Validation accuracy is compared to measure performance.


