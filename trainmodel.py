import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("student_grades_1000_with_all_required_grades.csv")

# Subject list
subjects = ["Maths", "Physics", "Biology", "Social", "Telugu", "Hindi", "English"]

# Create presence indicators and convert marks
for subject in subjects:
    df[f"{subject}_Present"] = df[subject].apply(lambda x: 0 if x == "AB" else 1)
    df[subject] = pd.to_numeric(df[subject], errors='coerce')

# Grade assignment
def calculate_grade(row):
    if any(pd.isna(row[subj]) or row[subj] < 35 for subj in subjects):
        return "No Grade"
    total = sum(row[subj] for subj in subjects)
    percentage = (total / 700) * 100
    if 91 <= percentage <= 100:
        return "A1"
    elif 81 <= percentage <= 90:
        return "A2"
    elif 71 <= percentage <= 80:
        return "B1"
    elif 61 <= percentage <= 70:
        return "B2"
    elif 51 <= percentage <= 60:
        return "C"
    elif 40 <= percentage <= 50:
        return "D"
    elif 0 <= percentage < 40:
        return "E"
    else:
        return "No Grade"

df["Grade"] = df.apply(calculate_grade, axis=1)

# Remove invalid rows
df_clean = df[df["Grade"] != "No Grade"].dropna()

# Features and labels
X = df_clean[subjects + [f"{s}_Present" for s in subjects]]
y = df_clean["Grade"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "grade_predictor_presence_model.pkl")
joblib.dump(le, "label_encoder_presence.pkl")

# Evaluate and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
