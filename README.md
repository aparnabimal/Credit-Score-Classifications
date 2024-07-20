# Credit Score Classification Project

This project aims to develop and evaluate various machine learning models for classifying credit scores based on customer financial and behavioral data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
    - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Feature Engineering](#feature-engineering)
    - [Model Training and Evaluation](#model-training-and-evaluation)
- [Models](#models)
    - [Multi-layer Feedforward Neural Network (Base Model)](#multi-layer-feedforward-neural-network-base-model)
    - [Multi-layer Feedforward Neural Network with Regularization and Learning Rate Adjustment](#multi-layer-feedforward-neural-network-with-regularization-and-learning-rate-adjustment)
    - [Sequential Neural Network (No Fine-tuning)](#sequential-neural-network-no-fine-tuning)
    - [Sequential Neural Network (with Fine-tuning)](#sequential-neural-network-with-fine-tuning)
- [Results and conclusion](#results)
- [How to Use](#how-to-use)


## Project Overview

Accurately classifying credit scores is crucial for financial institutions to assess the creditworthiness of potential borrowers. This project aims to build robust models to predict credit scores (Good, Standard, Poor) based on various customer attributes, such as income, debt, payment behavior, and credit history.

## Dataset

* **Description:** [Briefly describe the dataset, including the number of samples, features, and target variable]


## Methodology

### Data Cleaning and Preprocessing

*   Removed irrelevant and sensitive information (e.g., ID, Customer_ID, Month, Name, SSN)
*   Handled missing values using KNN imputation
*   Converted categorical variables to numerical using one-hot encoding
*   Standardized numerical features for consistent scaling

### Exploratory Data Analysis (EDA)

*   Explored relationships between features and the target variable (credit score) through visualizations like bar charts, heatmaps, and scatter plots.
*   Identified potential correlations and patterns in the data to inform feature engineering and model selection.

### Feature Engineering

*   Derived new features from existing ones to potentially improve model performance (e.g., calculating debt-to-income ratio). 

### Model Training and Evaluation

*   Split the data into training and testing sets.
*   Trained and evaluated multiple neural network models:
    *   Multi-layer Feedforward Neural Network (Base Model)
    *   Multi-layer Feedforward Neural Network with Regularization and Learning Rate Adjustment
    *   Sequential Neural Network (No Fine-tuning)
    *   Sequential Neural Network (with Fine-tuning)
*   Used metrics like accuracy, precision, recall, and F1-score to assess model performance.
*   Compared the performance of different models to identify the best-performing one.

## Models

### Multi-layer Feedforward Neural Network (Base Model)

*   This model is a feedforward neural network with multiple hidden layers. It takes the preprocessed input features and passes them through a series of densely connected layers with ReLU activation functions. The final layer uses a softmax activation to produce probabilities for the three credit score classes (Good, Standard, Poor). The base model utilized the GlorotNormal weight initializer for improved convergence during training. The number of layers and the number of units per layer were determined through hyperparameter tuning using Keras Tuner.

### Multi-layer Feedforward Neural Network with Regularization and Learning Rate Adjustment

*   This model extends the base model by introducing L2 regularization to the Dense layers. L2 regularization helps prevent overfitting by adding a penalty term to the loss function that discourages large weights. Additionally, the learning rate of the Adam optimizer was made tunable to explore a wider range of learning rates and potentially find a more optimal value.

### Sequential Neural Network (No Fine-tuning)

*   This model is a simpler sequential neural network with several dense layers, each followed by a Dropout layer to mitigate overfitting. The final layer is a Dense layer with 3 units and softmax activation for multi-class classification. The number of layers, units per layer, and dropout rates were determined through experimentation, but not through rigorous hyperparameter tuning. Class weights were used to address class imbalance issues in the dataset.

### Sequential Neural Network (with Fine-tuning)

*   This model builds upon the previous sequential model by adding Batch Normalization layers after each Dense layer. Batch Normalization helps stabilize and accelerate the training process by normalizing the activations of the previous layer. Additional dense layers were also added to increase the model's capacity to learn.

## Results and conclusion

*   The four models seem to have similar overall performance, as their accuracy scores are all around 0.70-0.75. However, when looking at the precision, recall, and f1-score values for each class, some differences can be observed.
Model 1 and Model 2 have exactly the same performance, as their confusion matrices and classification reports are identical. These models have good accuracy scores of 0.75 and perform relatively well across all three classes. They have precision, recall, and f1-score values ranging from 0.70 to 0.77, indicating that they can correctly identify the different classes with relatively high accuracy.
Model 3 has a slightly lower accuracy score of 0.70, but it still performs relatively well. However, it appears to struggle more with classifying the first class, as its precision, recall, and f1-score values for this class are noticeably lower than for the other two classes. This means that it is less accurate in correctly identifying the first class compared to the other two classes.
Model 4 also has an accuracy score of 0.74 and performs similarly to Model 3 in terms of precision, recall, and f1-score values for the first and third classes. However, it has a higher precision, recall, and f1-score for the second class, indicating that it is more accurate in identifying this class compared to the other models.


## How to Use

1.  Clone this repository.
2.  Install the required dependencies.
3.  Run the Jupyter notebooks in the `notebooks/` directory.

