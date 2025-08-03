import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Connect to the feature store
store = FeatureStore(repo_path="feature_repo")

# 2. Create the entity DataFrame
# This DataFrame contains the entities and timestamps for which we want features.
# It must also contain the target variable for training.
entity_df = pd.read_parquet("feature_repo/data/housing_features.parquet")[[
    "event_timestamp",
    "house_id",
    "MedHouseVal"
]]

# 3. Retrieve historical features to build the training dataset
print("Retrieving training data from Feast...")
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "location_features:MedInc",
        "location_features:HouseAge",
        "location_features:AveBedrms",
        "location_features:Latitude",
    ],
).to_df()

print("\n--- Training Data ---")
print(training_data.head())

# 4. Train a model
print("\n--- Training Model ---")

# Define features (X) and target (y)
X = training_data[['MedInc', 'HouseAge', 'AveBedrms', 'Latitude']]
y = training_data['MedHouseVal']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"\nModel R^2 score: {score:.4f}")
