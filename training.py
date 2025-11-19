import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("corn_crop_dataset.csv")

# Use only crop stage as input
X = df[["crop stage"]]
y = df["crop disease"]

# Label encode stage and disease
enc_stage = LabelEncoder()
X["crop stage"] = enc_stage.fit_transform(X["crop stage"])

enc_disease = LabelEncoder()
y = enc_disease.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(enc_stage, open("stage_encoder.pkl", "wb"))
pickle.dump(enc_disease, open("disease_encoder.pkl", "wb"))

print("✔ Model trained and saved successfully!")
