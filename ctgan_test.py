import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder

# Create a sample dataset similar to the Adult Income dataset
sample_data = pd.DataFrame({
    "Age": [39, 50, 38, 53, 28, 37, 49, 31, 42, 35],
    "Workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private", "Private",
                  "Private", "Private", "Federal-gov", "Private", "Private"],
    "Fnlwgt": [77516, 83311, 215646, 234721, 338409, 284582, 160187, 164174, 159449, 186374],
    "Education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors",
                  "Masters", "9th", "Masters", "Bachelors", "Some-college"],
    "Education-Num": [13, 13, 9, 7, 13, 14, 5, 14, 13, 10],
    "Marital Status": ["Never-married", "Married-civ-spouse", "Divorced",
                       "Married-civ-spouse", "Married-civ-spouse",
                       "Married-civ-spouse", "Married-spouse-absent",
                       "Married-civ-spouse", "Married-civ-spouse", "Married-civ-spouse"],
    "Occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners",
                   "Handlers-cleaners", "Prof-specialty", "Exec-managerial",
                   "Other-service", "Prof-specialty", "Exec-managerial", "Exec-managerial"],
    "Relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband",
                     "Wife", "Wife", "Unmarried", "Husband", "Husband", "Husband"],
    "Race": ["White", "White", "White", "Black", "Black", "White", "Black", "White", "White", "White"],
    "Sex": ["Male", "Male", "Male", "Male", "Female", "Female", "Female", "Male", "Male", "Male"],
    "Capital Gain": [2174, 0, 0, 0, 0, 0, 0, 14084, 5178, 0],
    "Capital Loss": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Hours per week": [40, 13, 40, 40, 40, 40, 16, 50, 40, 40],
    "Native Country": ["United-States", "United-States", "United-States", "United-States",
                       "Cuba", "United-States", "Jamaica", "United-States",
                       "United-States", "United-States"],
    "Income": ["<=50K", "<=50K", "<=50K", "<=50K", "<=50K", ">50K", "<=50K", ">50K", ">50K", ">50K"]
})

# Save the dataset as CSV
file_path = "adult_income_sample.csv"
sample_data.to_csv(file_path, index=False)


# Provide download link
file_path

# Load dataset (replace 'your_dataset.csv' with actual path)
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

# Print column names and types
print(data.info())

# Identify categorical columns
categorical_columns = ["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Native Country", "Income"]

# Convert categorical columns using Label Encoding
le_dict = {}  # Store encoders for later decoding
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le  # Save encoder for later decoding

# Initialize the CTGAN model
ctgan = CTGAN(epochs=300, batch_size=500)

# Train CTGAN
ctgan.fit(data, categorical_columns)

# Generate 10 synthetic rows
synthetic_data = ctgan.sample(10)

# Display the generated synthetic data
print(synthetic_data)

synthetic_data.to_csv("synthetic_data.csv", index=False)
print("Synthetic data saved as synthetic_data.csv")

# Decode categorical columns
for col in categorical_columns:
    synthetic_data[col] = le_dict[col].inverse_transform(synthetic_data[col])

# Display decoded synthetic data
print(synthetic_data.head())


