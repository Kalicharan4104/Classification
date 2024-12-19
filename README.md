# MNIST Handwritten Digit Classification

This project involves building and training a neural network model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The dataset contains 70,000 grayscale images of handwritten digits, each of size 28x28 pixels, divided into training and testing sets.

---

## Project Overview

1. **Dataset:** MNIST
   - Train set: 60,000 images
   - Test set: 10,000 images

2. **Goal:** Predict the digit (0-9) from input images.

3. **Model Architecture:**
   - Input Layer: Flattened 28x28 pixels into a 1D array of size 784.
   - Hidden Layer: Dense layer with 100 neurons and ReLU activation.
   - Output Layer: Dense layer with 10 neurons (softmax activation) for classification.

4. **Training Details:**
   - Loss Function: Sparse Categorical Crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy
   - Epochs: 25
   - Validation Split: 20%

---

## Steps to Reproduce

1. **Load Libraries and Dataset**
   ```python
   import tensorflow as tf
   from tensorflow.keras import Sequential
   from tensorflow.keras.layers import Dense, Flatten
   import matplotlib.pyplot as plt
   import numpy as np
   from tensorflow.keras.datasets import mnist

   (train_x, train_y), (test_x, test_y) = mnist.load_data()
   ```

2. **Data Exploration**
   - Check dataset shape and visualize sample images.
   - Example:
     ```python
     plt.imshow(train_x[5], cmap='gray')
     plt.title(f'Label: {train_y[5]}')
     plt.show()
     ```

3. **Data Preprocessing**
   - Normalize the pixel values to the range [0, 1].
     ```python
     X_train = train_x / 255.0
     X_test = test_x / 255.0
     ```

4. **Model Construction**
   ```python
   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(100, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```

5. **Model Compilation**
   ```python
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   ```

6. **Model Training**
   ```python
   history = model.fit(X_train, train_y, epochs=25, validation_split=0.2)
   ```

7. **Model Evaluation**
   - Calculate accuracy and visualize predictions:
     ```python
     y_prob = model.predict(X_test)
     y_predict = y_prob.argmax(axis=1)
     ```
   - Evaluate model:
     ```python
     from sklearn.metrics import accuracy_score
     accuracy = accuracy_score(test_y, y_predict)
     print(f'Test Accuracy: {accuracy * 100:.2f}%')
     ```

8. **Confusion Matrix**
   ```python
   from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
   cm = confusion_matrix(test_y, y_predict)
   disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
   disp.plot(cmap=plt.cm.Blues)
   plt.show()
   ```

9. **Results Visualization**
   - Plot accuracy and loss:
     ```python
     plt.plot(history.history['accuracy'], label='Train Accuracy')
     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
     plt.legend()
     plt.show()
     ```
   - Predict and visualize:
     ```python
     plt.imshow(X_test[0], cmap='gray')
     plt.title(f'Predicted: {y_predict[0]}')
     plt.axis('off')
     plt.show()
     ```

---

## Key Metrics

- **Final Test Accuracy:** 97.27%

---

## Dependencies

- Python
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Future Improvements

- Implement additional hidden layers for improved accuracy.
- Experiment with different activation functions.
- Use advanced architectures like CNNs for better feature extraction.
- Perform hyperparameter tuning.

---

## Contact
For any queries or suggestions, feel free to contact the project maintainer.

