# Training a neural network with the MNIST database of 28x28 hand-written digits to read hand-written numbers
---
`tf.keras.utils.normalize(data, axis)` normalizes data in the tensor (multidimensional array) to make them between 0 and 1

`relu` = Rectify Linear. It is an activation function

In `sparse_categorical_crossentropy`, **`sparse`** could also be **`binary`** for cats vs. dogs for example


```python
import tensorflow as tf

import logging, sys
logging.disable(sys.maxsize) # Hide INFO messages

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Loads data into a new tensor (multidimensional array)

x_train = tf.keras.utils.normalize(x_train, axis=1) # Normalizes data and scales from 0 to 1
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # Add input layer (and flatten)

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Add first hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Add second hidden layer

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Add output layer. Softmax activation function not relu

model.compile(optimizer='adam',                      # Use the 'adam' optimizer. Default go-to optimizer in Keras
             loss='sparse_categorical_crossentropy', # NN's don't maximaze accuracy, they minimize loss
             metrics=['accuracy'])                   # Track accuracy

model.fit(x_train, y_train, epochs=3) 
```

    Epoch 1/3
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2647 - accuracy: 0.9231
    Epoch 2/3
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1076 - accuracy: 0.9668
    Epoch 3/3
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0738 - accuracy: 0.9775
    




    <keras.callbacks.History at 0x141b8957e20>



### As we can see after running, the network identifies hand-written digits with approximately 97% accuracy.
---
So, did the neural network really generalize, learn patterns, and learn the attributes of hand-written digits? Or did it simply memorize images within the sample?

`model.evaulate(x_test, y_test)` evaluates the network ***out of sample***. We can conclude that the network does successfully identify foreign images, but with a slightly lower accuracy and slightly higher loss.

A high delta between ***in sample*** and ***out of sample*** evaulation suggests the network is *"Over Fitting"*.



```python
val_loss, val_acc = model.evaluate(x_test, y_test) # validation loss and validation accuracy
print("Validation loss: ", val_loss, '(', round(val_loss*100, 2), '%)')
print("Validation accuracy: ", val_acc, '(', round(val_acc*100, 2), '%)')
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.1021 - accuracy: 0.9698
    Validation loss:  0.10213526338338852 ( 10.21 %)
    Validation accuracy:  0.9697999954223633 ( 96.98 %)
    

### Saving and loading models:


`model.save('owens_digit_recognizer.model')`\
`new_model = tf.keras.models.load_model('owens_digit_recognizer.model')`
