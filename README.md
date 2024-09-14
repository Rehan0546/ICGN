# Integrated Contextual Gate Network (ICGN) for fNIRS Brain-Computer Interface Applications

This repository contains the implementation of the **Integrated Contextual Gate Network (ICGN)** layer, designed to enhance classification accuracy in deep learning models, particularly for functional Near-Infrared Spectroscopy (fNIRS) brain–computer interface (BCI) applications. This custom TensorFlow layer is inspired by the paper:

**"Enhancing Classification Accuracy with Integrated Contextual Gate Network: Deep Learning Approach for Functional Near-Infrared Spectroscopy Brain–Computer Interface Application"**  
*Sensors 2024, 24(10), 3040*  
[Link to Paper](https://doi.org/10.3390/s24103040)  
By: *Jamila Akhter, Noman Naseer, Hammad Nazeer, Haroon Khan, and Peyman Mirtaheri*

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Adding ICGN to Your Model](#adding-icgn-to-your-model)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
- [Citation](#citation)
- [License](#license)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/icgn-layer.git
   cd icgn-layer
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow
   ```

## Usage

### Adding ICGN to Your Model

You can add the **ICGN** layer to your TensorFlow model as follows:

```python
import tensorflow as tf
from icgn_layer import ICGN  # Assuming icgn_layer.py is the file name

# Initialize the model
model = tf.keras.Sequential()

# Add the custom ICGN layer
model.add(ICGN(num_hiddens=36, kernel_regularizer=tf.keras.regularizers.l2(0.01)))

# Add other layers as needed
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Saving and Loading the Model

To save the model after training, you can simply use TensorFlow's `model.save()` method:

```python
# Save the model
model.save('path_to_your_model')
```

When loading the saved model with the custom `ICGN` layer, you need to pass the custom object during the loading process:

```python
# Load the model with the custom ICGN layer
loaded_model = tf.keras.models.load_model('path_to_your_model', custom_objects={'ICGN': ICGN})
```

### Example of Full Workflow

```python
# Load necessary modules
import tensorflow as tf
from icgn_layer import ICGN

# Build the model
model = tf.keras.Sequential([
    ICGN(36, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train your model on the dataset (X_train, y_train)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save('my_icgn_model')

# Load the model later for inference or further training
loaded_model = tf.keras.models.load_model('my_icgn_model', custom_objects={'ICGN': ICGN})
```

## Citation

If you use this code, please cite the following paper:

Akhter, J., Naseer, N., Nazeer, H., Khan, H., & Mirtaheri, P. (2024). *Enhancing Classification Accuracy with Integrated Contextual Gate Network: Deep Learning Approach for Functional Near-Infrared Spectroscopy Brain–Computer Interface Application*. Sensors, 24(10), 3040. https://doi.org/10.3390/s24103040

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
