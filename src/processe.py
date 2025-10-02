import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ========== 1. Load Data ==========
df = pd.read_csv("data/processed/cleaned_data.csv")

# Split features/target
X = df.drop(columns=["charges"])
y = df["charges"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=474
)

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)