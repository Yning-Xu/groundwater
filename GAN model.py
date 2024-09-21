#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import pandas as pd
import numpy as np


# In[ ]:


data = df.values 
X, Y = data[:, :35], data[:, 35:]
X = X.astype('float32')
Y = Y.astype('float32')

X_normalized = scaler1.fit_transform(X)
Y_normalized = scaler2.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, train_size=0.85, random_state)


# In[ ]:


def build_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.35))
    model.add(Dense(output_dim, activation= 'sigmoid'))
    return model

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(Dense(1, activation= 'sigmoid'))
    return model

generator = build_generator(35, 8)
discriminator = build_discriminator(8)

generator_optimizer =  tf.keras.optimizers.Adam(learning_rate)
generator.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator.compile(optimizer= discriminator_optimizer, loss='binary_crossentropy')


# In[ ]:


for epoch in range(epochs):
    for batch_index in range(0, len(X_train), batch_size):
        
        X_batch = X_train[batch_index:batch_index+batch_size]
        Y_batch = Y_train[batch_index:batch_index+batch_size]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(X_batch, training=True)

            d_loss_real = discriminator(Y_batch, training=True)
            d_loss_fake = discriminator(generated_data, training=True)
            d_loss = 0.5 * (tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(d_loss_real),d_loss_real,from_logits=False))+
                           tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(d_loss_fake),d_loss_fake,from_logits=False)))
            
            adversarial_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(d_loss_fake), d_loss_fake,from_logits=False))
            difference_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(Y_batch, generated_data))
            gen_loss = a * adversarial_loss + b * difference_loss
            

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
          
generator.save()
discriminator.save()


# In[ ]:


Y_realpre = generator.predict(Obs)
best_value = np.array(Y_realpre).reshape(1, 8)
M= scaler2.inverse_transform(np.array(best_value))

relative_error = 100*np.abs((xref - M) / (xref + 1e-8))  
relative_error_flat = relative_error.flatten()

relative_error_list = []

for error in relative_error_flat:
    relative_error_list.append("{:.2f}".format(error))
    
def relative_error_vector(a, b):
    return np.abs(a - b) / np.abs(a) * 100

error_value = relative_error_vector(obs, e)

