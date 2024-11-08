# Detecting COVID-19 with Chest X-Ray using PyTorch

Image classification of chest X-rays into three classes: **Normal**, **Viral Pneumonia**, and **COVID-19**. This project uses a deep learning model to analyze chest X-ray images for COVID-19 detection, leveraging the [COVID-19 Radiography Dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) on Kaggle.

## Dataset
The dataset used in this project is sourced from Kaggle and contains X-ray images labeled as Normal, Viral Pneumonia, or COVID-19, which are used to train and test the model.

## Installation and Requirements
To run this project, you need the following libraries:

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `matplotlib`
- `PIL`
- `os`
- `random`
- `shutil`

## Project Structure

The notebook covers the following sections:

1. **Preparing Training and Test Sets**: Organizing the dataset into appropriate training and testing directories.
2. **Creating Custom Dataset**: Setting up a PyTorch `Dataset` class for handling data loading.
3. **Image Transformations**: Applying data augmentations and transformations.
4. **DataLoader Preparation**: Configuring data loaders for training and validation.
5. **Data Visualization**: Displaying sample images and transformations.
6. **Model Creation**: Building a convolutional neural network (CNN) model with PyTorch.
7. **Model Training**: Training the model on the dataset and monitoring performance.
8. **Final Results**: Evaluating the model’s performance and visualizing results.

## Usage

1. **Prepare the Dataset**: Download the dataset from Kaggle and organize it into the specified folders as described in the notebook.
2. **Run the Notebook**: Execute each cell to preprocess the data, create the model, and train it on the dataset.
3. **Model Evaluation**: After training, the notebook includes code for evaluating the model’s accuracy on test images.

## Results

The final section of the notebook includes performance metrics and visualizations that highlight the model’s accuracy in detecting COVID-19, viral pneumonia, and normal cases.
