# Image Classification with CNN using Keras

# Task 1: Import Libraries


```python
!pip install tensorflow
```

    ... logs for installation ...
    


```python
import tensorflow as tf
import os
import numpy as np

from matplotlib import pyplot as plt
%matplotlib inline

if not os.path.isdir('models'):
    os.mkdir('models')
    
print('TensorFlow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())
```

    TensorFlow version: 2.8.0
    Is using GPU? False
    

# Task 2: Preprocess Data


```python
def get_three_classes(x, y):
    indices_0, _ = np.where(y == 0.)
    indices_1, _ = np.where(y == 1.)
    indices_2, _ = np.where(y == 2.)

    indices = np.concatenate([indices_0, indices_1, indices_2], axis=0)
    
    x = x[indices]
    y = y[indices]
    
    count = x.shape[0]
    indices = np.random.choice(range(count), count, replace=False)
    
    x = x[indices]
    y = y[indices]
    
    y = tf.keras.utils.to_categorical(y)
    
    return x, y
```


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("Train whole set:")
print(x_train.shape, y_train.shape)
print("Test whole set:")
print(x_test.shape, y_test.shape)

x_train, y_train = get_three_classes(x_train, y_train)
x_test, y_test = get_three_classes(x_test, y_test)
print("Train subset:")
print(x_train.shape, y_train.shape)
print("Test subset:")
print(x_test.shape, y_test.shape)
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 415s 2us/step
    170508288/170498071 [==============================] - 415s 2us/step
    Train whole set:
    (50000, 32, 32, 3) (50000, 1)
    Test whole set:
    (10000, 32, 32, 3) (10000, 1)
    Train subset:
    (15000, 32, 32, 3) (15000, 3)
    Test subset:
    (3000, 32, 32, 3) (3000, 3)
    

# Task 3: Visualize Examples


```python
class_names = ['aeroplane', 'car', 'bird']

def show_random_examples(x, y, p):
    indices = np.random.choice(range(x.shape[0]), 10, replace=False)
    
    x = x[indices]
    y = y[indices]
    p = p[indices]
    
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, 1 + i)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        plt.xlabel(class_names[np.argmax(p[i])], color=col)
    plt.show()
    
show_random_examples(x_train, y_train, y_train)
```


    
![png](images/output_8_0.png)
    



```python
show_random_examples(x_test, y_test, y_test)
```


    
![png](images/output_9_0.png)
    


# Task 4: Create Model


```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense

def create_model():
    def add_conv_block(model, num_filters):
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model

    model = tf.keras.models.Sequential()
    model.add(Input(shape=(32, 32,3)))
    
    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)
    
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy']
    )
    return model

model = create_model()
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 32, 32, 32)        896       
                                                                     
     batch_normalization (BatchN  (None, 32, 32, 32)       128       
     ormalization)                                                   
                                                                     
     conv2d_1 (Conv2D)           (None, 30, 30, 32)        9248      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0         
     )                                                               
                                                                     
     dropout (Dropout)           (None, 15, 15, 32)        0         
                                                                     
     conv2d_2 (Conv2D)           (None, 15, 15, 64)        18496     
                                                                     
     batch_normalization_1 (Batc  (None, 15, 15, 64)       256       
     hNormalization)                                                 
                                                                     
     conv2d_3 (Conv2D)           (None, 13, 13, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0         
     2D)                                                             
                                                                     
     dropout_1 (Dropout)         (None, 6, 6, 64)          0         
                                                                     
     conv2d_4 (Conv2D)           (None, 6, 6, 128)         73856     
                                                                     
     batch_normalization_2 (Batc  (None, 6, 6, 128)        512       
     hNormalization)                                                 
                                                                     
     conv2d_5 (Conv2D)           (None, 4, 4, 128)         147584    
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 2, 2, 128)        0         
     2D)                                                             
                                                                     
     dropout_2 (Dropout)         (None, 2, 2, 128)         0         
                                                                     
     flatten (Flatten)           (None, 512)               0         
                                                                     
     dense (Dense)               (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 289,443
    Trainable params: 288,995
    Non-trainable params: 448
    _________________________________________________________________
    

# Task 5: Train the Model


```python
h = model.fit(
    x_train/255., y_train,
    validation_data=(x_test/255., y_test),
    epochs=20, batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            'models/model_{val_accuracy:.3f}.h5',
            save_best_only=True, save_weights_only=False,
            monitor='val_accuracy'
        )
    ]
)
```

    Epoch 1/20
    235/235 [==============================] - 21s 90ms/step - loss: 0.2868 - accuracy: 0.8900 - val_loss: 0.2514 - val_accuracy: 0.9090
    Epoch 2/20
    235/235 [==============================] - 21s 89ms/step - loss: 0.2700 - accuracy: 0.8978 - val_loss: 0.6099 - val_accuracy: 0.8137
    Epoch 3/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2534 - accuracy: 0.9031 - val_loss: 0.2670 - val_accuracy: 0.9060
    Epoch 4/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2441 - accuracy: 0.9059 - val_loss: 0.3108 - val_accuracy: 0.8820
    Epoch 5/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2371 - accuracy: 0.9115 - val_loss: 0.2478 - val_accuracy: 0.9137
    Epoch 6/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2228 - accuracy: 0.9144 - val_loss: 0.2014 - val_accuracy: 0.9240
    Epoch 7/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2144 - accuracy: 0.9172 - val_loss: 0.3750 - val_accuracy: 0.8747
    Epoch 8/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2089 - accuracy: 0.9224 - val_loss: 0.3351 - val_accuracy: 0.8707
    Epoch 9/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.2082 - accuracy: 0.9197 - val_loss: 0.2159 - val_accuracy: 0.9150
    Epoch 10/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.1913 - accuracy: 0.9283 - val_loss: 0.2120 - val_accuracy: 0.9297
    Epoch 11/20
    235/235 [==============================] - 21s 87ms/step - loss: 0.1879 - accuracy: 0.9292 - val_loss: 0.3387 - val_accuracy: 0.8870
    Epoch 12/20
    235/235 [==============================] - 21s 88ms/step - loss: 0.1843 - accuracy: 0.9303 - val_loss: 0.1869 - val_accuracy: 0.9257
    Epoch 13/20
    235/235 [==============================] - 21s 89ms/step - loss: 0.1764 - accuracy: 0.9347 - val_loss: 0.2107 - val_accuracy: 0.9210
    Epoch 14/20
    235/235 [==============================] - 20s 87ms/step - loss: 0.1751 - accuracy: 0.9338 - val_loss: 0.1792 - val_accuracy: 0.9340
    Epoch 15/20
    235/235 [==============================] - 20s 87ms/step - loss: 0.1727 - accuracy: 0.9343 - val_loss: 0.1792 - val_accuracy: 0.9360
    Epoch 16/20
    235/235 [==============================] - 20s 87ms/step - loss: 0.1650 - accuracy: 0.9353 - val_loss: 0.2351 - val_accuracy: 0.9173
    Epoch 17/20
    235/235 [==============================] - 20s 87ms/step - loss: 0.1594 - accuracy: 0.9394 - val_loss: 0.1770 - val_accuracy: 0.9337
    Epoch 18/20
    235/235 [==============================] - 20s 86ms/step - loss: 0.1582 - accuracy: 0.9393 - val_loss: 0.1861 - val_accuracy: 0.9297
    Epoch 19/20
    235/235 [==============================] - 20s 86ms/step - loss: 0.1504 - accuracy: 0.9437 - val_loss: 0.1990 - val_accuracy: 0.9273
    Epoch 20/20
    235/235 [==============================] - 21s 87ms/step - loss: 0.1521 - accuracy: 0.9422 - val_loss: 0.1856 - val_accuracy: 0.9333
    

# Task 6: Final Predictions


```python
accs = h.history['accuracy']
val_accs = h.history['val_accuracy']

plt.plot(range(len(accs)), accs, label='Training')
plt.plot(range(len(accs)), val_accs, label='Validation')
plt.legend()
plt.show()
```


    
![png](images/output_15_0.png)
    



```python
model= tf.keras.models.load_model('models/model_0.936.h5')
```


```python
preds = model.predict(x_test/255.)
```


```python
show_random_examples(x_test, y_test, preds)
```


    
![png](images/output_18_0.png)
    



```python

```
