# Pawpularity-CI

This repository contains the code for the Minor Project of CS354N, Group 16.

## Description

The goal of this project is to develop a system for measuring the "Pawpularity" of the images of Cats and Dogs based on the images and metadata like eyes, collage, human, near, etc.

## Installation

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/Popularity-CI.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. To run the streamlit interface, run `streamlit run app.py`
4. The dataset is publically available at: `https://www.kaggle.com/competitions/petfinder-pawpularity-score/data`
5. The Experiments folder contains the different experiments we have run as stated in the report. Please refer to the report while viewing the Experiments folder.
6. The streamlit interface uses the weights of `/Experiments/Sample_weighted_loss_2.5_and_2.ipynb` model. This model has given the best generalised performance.
7. Find the training logs and output of all the experiments at: `https://www.kaggle.com/code/rohanjha04/pawpularity-ci`. Click on `versions` to get Weights for different experiments along with the training and testing logs.

## Usage

Once the project is up and running, you can perform the following actions:

- View the pawpularity metrics of a specific image.
- Upload the image and select the metadata with help of checkbox.
- Keep the weights in the same folder as `app.py`. Else update the correct path to the weights. Similarly update the path to the dataset if you are re-running any experiments.

## Contributers
1. Jha Rohan
2. Hrishesh Sharma
3. Nishkarsh Luthra
