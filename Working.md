# Nuclear Power Plant Accident Classification Guide

## What Does This Project Do?

Imagine you're working at a nuclear power plant. Suddenly, alarms start blaring - something's gone wrong! But what exactly is happening? Is it a cooling problem? A pressure issue? A broken pipe?

This project helps answer that crucial question. It's like a detective that looks at all the sensor readings from the plant (temperatures, pressures, flow rates, etc.) and figures out what type of accident is occurring. Knowing the type of accident quickly can save lives by helping operators take the right emergency actions.

## The Story of How It Works

### Where Does Our Data Come From?

Nuclear power plants have sensors everywhere - measuring water temperature, pressure, radiation levels, and dozens of other things. When something goes wrong, these sensors recorded how values change over time.

For this project, we have data from simulated accidents (nobody wants to create real nuclear accidents just for data!). This data is organized in folders based on the type of accident:

```
Our folders:
├── Operation_csv_data/   (contains all the sensor readings during accidents)
│   ├── ATWS/             (one type of accident)
│   │   ├── 1.csv         (data from the first simulation)
│   │   ├── 2.csv         (data from the second simulation)
│   │   └── ...
│   ├── LOCA/             (another type of accident)
│   │   ├── 1.csv
│   │   └── ...
│   └── ...               (other accident types)
├── Dose_csv_data/        (radiation measurements)
└── NPPAD/                (extra accident information)
```

### Let's Look Inside a Data File

Let's open up one of these files - for example, `Operation_csv_data/ATWS/1.csv`. This file shows what happens during an "Anticipated Transient Without Scram" accident (that's a scary one where the emergency shutdown system fails).

The file looks something like this:

```
TIME,P,TAVG,THA,THB,TCA,...
0.0,155.500015,310.000000,327.823975,327.823212,292.176056,...
10.0,155.473068,309.983185,327.808258,327.807434,292.159302,...
20.0,155.445129,309.966217,327.791992,327.791229,292.142273,...
...
```

What does this mean? Let's break it down:

- `TIME`: Seconds since the accident started
- `P`: Pressure in the reactor (155.5 units at the start)
- `TAVG`: Average temperature (310 units at the start)
- `THA`, `THB`: Hot leg temperatures (327.8 units at the start)
- `TCA`: Cold leg temperature (292.1 units at the start)
- ...and about 90 other measurements!

So we can see that 10 seconds into this ATWS accident, the pressure dropped slightly from 155.50 to 155.47, and the average temperature dropped from 310.00 to 309.98.

### Our Challenge

We have 17 different types of accidents:

1. **ATWS** - Anticipated Transient Without Scram (emergency shutdown fails)
2. **FLB** - Feedwater Line Break (pipe carrying water to the steam generator breaks)
3. **LOCA** - Loss of Coolant Accident (coolant leaks from the reactor)
4. **SGBTR** - Steam Generator B Tube Rupture
5. ...and 13 others!

Each accident affects the sensor readings differently. Our job is to create a computer program that can look at these readings and identify which type of accident is happening.

### How We Solve This Problem - Step by Step

#### 1. Collecting the Data

First, we gather all the accident data files. Our program searches through all the folders and finds files like `ATWS/1.csv`, `LOCA/1.csv`, etc. For each file, it reads all the sensor values over time.

```python
# This code finds all our accident types
accident_types = [d for d in os.listdir(OPERATION_DATA_DIR)
                 if os.path.isdir(os.path.join(OPERATION_DATA_DIR, d)) and d != 'Normal']

# Now we know we're looking at ATWS, LOCA, FLB, etc.
```

#### 2. Extracting Important Information

Looking at every single measurement at every second would be too much data for our program to handle effectively. So we calculate summary statistics that capture the important characteristics of each accident.

For example, with the pressure (P) during an ATWS accident:

- The average pressure: 155.51 units
- How much the pressure varies: 0.006 units
- The minimum pressure: 155.49 units
- The maximum pressure: 155.51 units
- The typical rate of change: -0.0027 units per second

We do this for all 97 sensor columns, creating hundreds of features that describe each accident.

```python
# For the pressure values in ATWS/1.csv
features = {
  'P_mean': 155.50999727797233,  # Average pressure
  'P_std': 0.0059739248639754106,  # How much pressure varied
  'P_min': 155.48999023437384,  # Lowest pressure
  'P_max': 155.51998901367188,  # Highest pressure
  'P_diff_mean': -0.0026922903230294693  # Average change in pressure
  # ...and hundreds more features
}
```

#### 3. Teaching Our Computer to Recognize Accidents

Now we use a type of artificial intelligence called a "Random Forest Classifier." Think of it like training many detectives, each looking for different clues, and then having them vote on what type of accident they think is happening.

We train our computer using most of our accident data (80%), keeping some aside (20%) to test if the computer has really learned to identify accidents correctly.

```
Training the system...
- Using 80% of accident data for training
- Testing with 20% to verify accuracy
```

#### 4. Measuring How Good Our Detector Is

After training, we test our accident detector and find it works extremely well:

```
Accuracy: 99.09% (based on 10-fold cross-validation)

Detailed accuracy for each accident type:
              precision    recall  f1-score   support
         FLB       1.00      1.00      1.00        20
         LLB       1.00      1.00      1.00        21
        LOCA       1.00      1.00      1.00        20
       LOCAC       1.00      1.00      1.00        20
          LR       1.00      1.00      1.00        20
          MD       0.95      1.00      0.98        20
       SGATR       1.00      1.00      1.00        20
       SGBTR       1.00      1.00      1.00        22
       SLBIC       1.00      1.00      1.00        20
       SLBOC       1.00      0.95      0.97        20
       ...
```

This means that for most types of accidents, our system correctly identifies them 100% of the time! Even for the trickier ones, it still gets them right at least 95% of the time.

#### 5. Finding What Clues Matter Most

We also discover which sensor readings are most important for identifying accidents:

```
Most important measurements for identifying accidents:
1. WFWB_max: 0.012356  (Maximum feedwater flow to steam generator B)
2. QMGA_max: 0.012302  (Maximum steam flow rate from generator A)
3. STTB_diff_std: 0.011322  (Variation in steam temperature changes in generator B)
4. QMWT_max: 0.010428  (Maximum total steam flow)
5. WSTA_max: 0.010277  (Maximum steam flow in generator A)
```

This tells us that steam and water flow rates are especially important in distinguishing different types of accidents.

### An Example: Detecting an ATWS Accident

Let's walk through how our system would detect an Anticipated Transient Without Scram (ATWS) accident:

1. **Get the sensor data**: We read the values from all sensors during the first 10 minutes of an incident.

2. **Calculate features**: We compute statistics for each sensor:

   ```
   P_mean: 155.50999727797233  (Average pressure)
   P_std: 0.0059739248639754106  (Pressure variation)
   TAVG_mean: 309.9921875  (Average temperature)
   TAVG_std: 0.0178125  (Temperature variation)
   ...and hundreds more
   ```

3. **Pass these features to our trained model**: The model evaluates all these values.

4. **Get the prediction**: The model responds with "ATWS" as the accident type, with 99.7% confidence.

5. **Alert operators**: The system tells nuclear plant operators that an ATWS accident is occurring, allowing them to implement the correct emergency procedures.

### How to Use This System Yourself

If you want to try this system, follow these steps:

1. **Install the required software**:

   ```
   pip install pandas numpy matplotlib scikit-learn joblib tqdm seaborn
   ```

2. **Train the model** (or use our pre-trained one):

   ```
   python nuclear_accident_classification.py
   ```

   This will read all the accident data, extract features, train the model, and save it.

3. **Use the model to predict accident types**:

   ```python
   # Load the model
   model = joblib.load("model_output/best_model.joblib")

   # Predict the accident type from new sensor data
   prediction = predict_accident_type(model, "new_sensor_data.csv", logger)

   # See what type of accident it is
   print(f"Accident type: {prediction['predicted_type']}")
   ```

### Behind the Scenes: How the Magic Works

While the concept is simple (read sensor data, calculate statistics, identify patterns), the actual process involves several clever techniques:

1. **Feature Engineering**: We transform raw sensor readings into meaningful statistics that capture the essence of each accident type.

2. **Random Forest Classification**: We use an ensemble of decision trees (like a committee of experts) to vote on the accident type.

3. **Cross-Validation**: We test our model on different subsets of data to ensure it works reliably.

4. **Hyperparameter Tuning**: We try different settings for our model to find the most accurate configuration.

5. **Feature Importance Analysis**: We identify which sensor readings matter most for accident classification.

### Why This Matters

In a real nuclear power plant, identifying the type of accident quickly can make the difference between a minor incident and a major disaster. Different accidents require different responses:

- A **Loss of Coolant Accident (LOCA)** might require immediate injection of emergency coolant
- A **Steam Generator Tube Rupture (SGTR)** might require isolating the affected steam generator
- An **Anticipated Transient Without Scram (ATWS)** might require manual insertion of control rods

By automatically identifying the accident type from sensor data, our system helps operators take the right actions faster, potentially saving lives and preventing environmental damage.

## Detailed Explanations of Techniques

### Feature Engineering

Feature engineering is the process of using domain knowledge to select and transform raw data into informative features that can be used in machine learning models. In the context of nuclear accident classification, feature engineering involves analyzing sensor data to extract meaningful statistics that capture the essence of each accident type. This includes calculating averages, standard deviations, and other statistical measures for each sensor reading. By doing so, we reduce the complexity of the data while retaining the most important information that helps in distinguishing different types of accidents.

For example, if we have pressure readings from a reactor, we might calculate:

- The average pressure over time: 155.51 units
- The variation in pressure (standard deviation): 0.006 units
- The minimum and maximum pressure values: 155.49 to 155.51 units
- The rate of change in pressure over time: -0.0027 units per second

These features help the model understand the typical behavior of the reactor during different types of accidents.

### Random Forest Classification

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their predictions. It is particularly effective for classification tasks because it reduces overfitting and improves accuracy. In our project, Random Forest is used to classify the type of nuclear accident based on the engineered features.

Each decision tree in the forest is trained on a random subset of the data, and the final prediction is made by aggregating the predictions of all trees. This approach leverages the "wisdom of the crowd" to make more accurate predictions. For instance, the model might predict an "ATWS" accident with 99.7% confidence.

### Cross-Validation

Cross-validation is a technique used to assess the performance of a machine learning model. It involves dividing the dataset into multiple subsets, training the model on some subsets, and validating it on the remaining ones. This process is repeated several times, and the results are averaged to provide a more reliable estimate of the model's performance.

In our project, we use 10-fold cross-validation, which means the data is divided into 10 parts. The model is trained on 9 parts and tested on the 10th part, and this process is repeated 10 times. This helps ensure that the model generalizes well to unseen data. The log file shows an accuracy of 99.09%.

### Hyperparameter Tuning

Hyperparameter tuning involves finding the optimal settings for a machine learning model's hyperparameters, which are parameters that are not learned from the data but set before the training process. For Random Forest, hyperparameters include the number of trees, the maximum depth of each tree, and the minimum number of samples required to split a node.

By experimenting with different hyperparameter values, we can improve the model's performance. This is typically done using techniques like grid search or random search, where various combinations of hyperparameters are tested to find the best configuration.

### Feature Importance Analysis

Feature importance analysis helps identify which features are most influential in making predictions. In Random Forest, this is done by measuring the decrease in prediction accuracy when a feature is removed. Features that cause a significant drop in accuracy are considered important.

Understanding feature importance is crucial for interpreting the model's decisions and ensuring that it relies on meaningful data. In our project, we found that certain sensor readings, such as steam and water flow rates, are particularly important for distinguishing different types of accidents. The log file highlights features like maximum feedwater flow and steam flow rates as critical for classification.

## Log File Explanation

The training log file provides a detailed account of the model's training process. It includes information about the data being processed, the features extracted, and the performance metrics at each step. Here are some key points from the log file:

- **Accident Types**: The log lists the different types of accidents being classified, such as ATWS, FLB, and LOCA.
- **Data Preview**: It shows a preview of the input data, including the first few rows and columns.
- **Feature Extraction**: The log details the features extracted from the data, such as mean and standard deviation for each sensor reading.
- **Training Process**: It records the progress of the training process, including the number of features used and the performance metrics achieved.

By analyzing the log file, we can gain insights into how the model is learning and identify any potential issues that need to be addressed.

## Feature Importance File Explanation

The `feature_importance.csv` file contains a list of features along with their importance scores, which indicate how much each feature contributes to the model's predictions. Here's a detailed explanation of its contents:

1. **Structure**:

   - The file is organized into two columns: `feature` and `importance`.
   - Each row represents a feature extracted from the sensor data, with its corresponding importance score.

2. **Features**:

   - The `feature` column lists the names of the features. These names are typically derived from the sensor readings and the statistical measures calculated during feature engineering. For example, `STTB_diff_std` represents the standard deviation of the difference in steam temperature changes in generator B.

3. **Importance Scores**:

   - The `importance` column provides a numerical score for each feature, indicating its relative importance in the model's decision-making process.
   - Higher scores suggest that the feature has a greater impact on the model's ability to classify accidents accurately.

4. **Top Features**:

   - The file is sorted by importance scores in descending order, so the most important features appear at the top.
   - For instance, `STTB_diff_std` has the highest importance score of 0.01588, making it a critical feature for the model.

5. **Interpretation**:
   - Understanding which features are most important helps in interpreting the model's predictions and ensuring that it relies on meaningful data.
   - It also provides insights into which sensor readings are most indicative of different types of nuclear accidents.
