# Created by Yanan Liu on 14:43 10/11/2023
# Location: Your Location
# Log: Your Log Information
# Version: Your Version Information

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load MNIST data
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target.astype(int)

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Echo State Network parameters
n_reservoir = 500
sparsity = 0.1
radius = 0.9

# Initialize random reservoir
np.random.seed(42)
W_reservoir = np.random.rand(n_reservoir, n_reservoir) - 0.5
W_reservoir[np.random.rand(*W_reservoir.shape) > sparsity] = 0
spectral_radius = max(abs(np.linalg.eigvals(W_reservoir)))
W_reservoir *= radius / spectral_radius

# Train reservoir
state = np.zeros(n_reservoir)
reservoir_states = []

for image in X_train:
    for pixel in image:
        state = np.tanh(np.dot(W_reservoir, state) + pixel)
    reservoir_states.append(state)

# Train logistic regression on reservoir states
clf = LogisticRegression()
clf.fit(reservoir_states, y_train)

# Test model
test_states = []
for image in X_test:
    state = np.zeros(n_reservoir)
    for pixel in image:
        state = np.tanh(np.dot(W_reservoir, state) + pixel)
    test_states.append(state)

predictions = clf.predict(test_states)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
