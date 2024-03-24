## FXCM Deep Learning-Based Trading Strategy

This repository contains code for developing, training, testing, and implementing a deep learning-based trading strategy using historical price data and real-time streaming data from FXCM markets. The strategy aims to predict price movements and generate buy/sell signals based on these predictions, with the ultimate goal of achieving profitable trading outcomes.

### Workflow Overview

1. **Loading and Preparing Data**:
   - Historical price data is loaded from a CSV file named "DNN_data.csv" into a pandas DataFrame.
   - Data visualization techniques are applied, and log returns of the price series are calculated.
   - Additional features such as the simple moving average (SMA), Bollinger bands (BOLL), minimum and maximum values, momentum (MOM), and volatility (VOL) are computed based on the historical price data.

2. **Adding Feature Lags**:
   - Lagged versions of the features are added to the DataFrame to capture temporal dependencies.

3. **Splitting Data into Train and Test Sets**:
   - The prepared DataFrame is split into training and testing datasets.

4. **Feature Scaling (Standardization)**:
   - The training data is standardized to have a mean of 0 and a standard deviation of 1.

5. **Creating and Fitting the DNN Model**:
   - A deep neural network (DNN) model is created and trained using the standardized features and the target variable (direction of price movement).

6. **Out-Sample Prediction and Forward Testing**:
   - The trained model is used to predict the direction of price movement for the test dataset.
   - Trading positions are determined based on the model predictions, and profit/loss calculations are performed.
   - The strategy's performance is evaluated using cumulative returns and compared with the buy-and-hold strategy.

7. **Saving Model and Parameters**:
   - The trained DNN model and the parameters used for standardization are saved for future use.

8. **Implementation**:
   - The trained model and parameters are loaded for implementation in a trading environment, either Oanda or FXCM.
   - A class (`DNNTrader`) is defined to handle the trading strategy, with platform-specific methods for interacting with the API.
   - The `DNNTrader` class implements methods for retrieving historical data, defining trading strategies, executing trades, and reporting trade results.

### DNNModule 

The `DNNModel.py` module provides a flexible framework for creating and configuring deep neural network models tailored to binary classification tasks. It allows users to customize various aspects of the model architecture and training process to suit the specific requirements of their classification problem. The `DNNModel.py` workflow can be summarized as follows:

1. **Setting Seeds (`set_seeds` function)**:
   - Seeds for random number generation in Python, NumPy, and TensorFlow are set to ensure reproducibility of results across multiple runs of the code.

2. **Class Weight Calculation (`cw` function)**:
   - Class weights for a binary classification problem are calculated based on the distribution of classes in the dataset. The function returns a dictionary mapping each class label to its corresponding weight.

3. **Legacy Adam Optimizer (`optimizer`)**:
   - The Adam optimizer with a specified learning rate is initialized. This optimizer is used to minimize the loss function during model training.

4. **Creating the Deep Neural Network Model (`create_model` function)**:
   - The function constructs a deep neural network (DNN) model using the Keras Sequential API.
   - It allows customization of various parameters such as the number of hidden layers, number of hidden units per layer, dropout regularization, activity regularization, regularization function (e.g., L1 or L2), and optimizer.
   - The model architecture consists of densely connected layers with ReLU activation functions.
   - Optional dropout regularization and activity regularization can be applied to prevent overfitting.
   - The final layer uses a sigmoid activation function for binary classification, predicting the probability of the positive class.
   - The model is compiled with binary cross-entropy loss and the specified optimizer, with accuracy as the evaluation metric.

### Usage
- Ensure you have the necessary dependencies installed (such as pandas, numpy, matplotlib, keras, etc.).
- Prepare your historical price data and ensure it is in the appropriate format.
- Run the provided code to train and test your deep learning-based trading strategy.
- Implement the strategy using the `DNNTrader` class 


# Guide to Running FXCM_Deep_Learning.ipynb

### Introduction
FXCM_Deep_Learning.ipynb is a Jupyter Notebook file that contains code for developing, training, testing, and implementing a deep learning-based trading strategy using historical price data and real-time streaming data from financial markets. This guide will walk you through the steps required to run the code successfully.

### Prerequisites
Before running the code, ensure that you have the following prerequisites installed:
1. Python (preferably version 3.8)
2. Jupyter Notebook
3. Necessary Python libraries (pandas, numpy, matplotlib, keras, fxcmpy)

### Steps to Run the Code

1. **Clone the Repository**:
   - Start by cloning or downloading the repository `FXCM_ML` to your local machine.

2. **Navigate to the Project Directory**:
   - Open a terminal or command prompt and navigate to the directory where you have cloned or downloaded the `FXCM_ML` repository.

3. **Install Dependencies**:
   - Install the necessary dependencies by running:
     ```
     pip install -r requirements.txt
     ```
   - This command will install all the required Python libraries specified in the `requirements.txt` file.

4. **Open Jupyter Notebook**:
   - Launch Jupyter Notebook by running the following command in the terminal:
     ```
     jupyter notebook
     ```
   - This will open Jupyter Notebook in your default web browser.

5. **Navigate to FXCM_Deep_Learning.ipynb**:
   - In Jupyter Notebook's file browser, navigate to the `FXCM_ML` directory and locate the file `FXCM_Deep_Learning.ipynb`.
   - Click on the file to open it in Jupyter Notebook.

6. **Execute the Code Cells**:
   - Once the notebook is open, you can run each code cell individually by selecting the cell and pressing `Shift + Enter`.
   - Follow the instructions provided in the code comments to load data, preprocess it, train the deep learning model, and implement the trading strategy.

7. **Provide Necessary Inputs**:
   - You may need to provide inputs such as file paths, API keys (if applicable), and trading parameters as specified in the code comments.
   - Ensure that you understand the inputs required for each code cell and provide them accordingly.

8. **Review the Outputs**:
   - After executing each code cell, review the outputs, visualizations, and any error messages that may appear.
   - Pay attention to any performance metrics or results generated by the trading strategy.

9. **Implement the Strategy**:
   - Once you have trained and tested the deep learning model successfully, you can proceed to implement the trading strategy in a live trading environment.
   - Follow the instructions provided in the code comments to implement the strategy using the `DNNTrader` class.

10. **Monitor and Evaluate**:
    - Monitor the performance of your trading strategy in the live trading environment and evaluate its effectiveness over time.
    - Make any necessary adjustments or optimizations based on the observed results.

### Disclaimer

- Running FXCM_Deep_Learning.ipynb requires a basic understanding of Python, Jupyter Notebook, and financial trading concepts. By following this guide and carefully executing each code cell, you can develop, train, test, and implement a deep learning-based trading strategy effectively. Remember to exercise caution and conduct thorough research before engaging in live trading activities.
- This code is provided for educational and informational purposes only. It does not constitute financial advice or trading recommendations.
- Trading in financial markets involves risk, and past performance is not indicative of future results. Always conduct thorough research and consider seeking advice from a qualified financial advisor before making any investment decisions.# FXCM_ML
# FXCM_ML
