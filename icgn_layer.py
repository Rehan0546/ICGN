# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:31:34 2024

@author: Jamila Akhtar
"""
 
import tensorflow as tf

class ICGN(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, kernel_regularizer, **kwargs):
        super(ICGN, self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.sigma = 0.01
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.num_inputs = input_shape[-1]

        # Add weight matrices and bias terms with regularization
        self.W_xi = self.add_weight(shape=(self.num_inputs, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.W_hi = self.add_weight(shape=(self.num_hiddens, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.b_i = self.add_weight(shape=(self.num_hiddens,),
                                   initializer=tf.keras.initializers.Zeros(),
                                   trainable=True)

        self.W_xf = self.add_weight(shape=(self.num_inputs, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.W_hf = self.add_weight(shape=(self.num_hiddens, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.b_f = self.add_weight(shape=(self.num_hiddens,),
                                   initializer=tf.keras.initializers.Zeros(),
                                   trainable=True)

        self.W_xo = self.add_weight(shape=(self.num_inputs, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.W_ho = self.add_weight(shape=(self.num_hiddens, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.b_o = self.add_weight(shape=(self.num_hiddens,),
                                   initializer=tf.keras.initializers.Zeros(),
                                   trainable=True)

        self.W_xc = self.add_weight(shape=(self.num_inputs, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.W_hc = self.add_weight(shape=(self.num_hiddens, self.num_hiddens),
                                    initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)
        self.b_c = self.add_weight(shape=(self.num_hiddens,),
                                   initializer=tf.keras.initializers.Zeros(),
                                   trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        H = tf.zeros((batch_size, self.num_hiddens))
        C = tf.zeros((batch_size, self.num_hiddens))
        outputs = []
        for X in tf.unstack(inputs, axis=1):
            '''
            I = tf.sigmoid(tf.matmul(X, self.W_xi) + tf.matmul(H, self.W_hi) + self.b_i)
            F = tf.sigmoid(tf.matmul(X, self.W_xf) + tf.matmul(H, self.W_hf) + self.b_f)
            O = tf.sigmoid(tf.matmul(X, self.W_xo) + tf.matmul(H, self.W_ho) + self.b_o)
            '''
            I = tf.sigmoid(tf.matmul(X, self.W_xi) + tf.matmul(C, self.W_hi) + tf.matmul(H, self.W_ho)+ self.b_i)
            F = tf.sigmoid(tf.matmul(X, self.W_xf) + tf.matmul(C, self.W_hf) + tf.matmul(H, self.W_ho)+ self.b_f)
            O = tf.sigmoid(tf.matmul(X, self.W_xo) + tf.matmul(C, self.W_ho) + tf.matmul(H, self.W_ho) + self.b_o)
            # C_tilde = tf.tanh(tf.matmul(X, self.W_xc) + tf.matmul(H, self.W_hc) + self.b_c)
            C_tilde = tf.tanh(tf.matmul(X, self.W_xc) + tf.matmul(H, self.W_hc) +  self.b_c) # Now cell state depending on input gate values
            C =  I*C_tilde  + F * C_tilde + O * C_tilde  # modified as adding O because cell state should be updated in according to output for next run



            H = O * tf.tanh(C)
            # H = O * tf.tanh(C)+ tf.matmul(C, self.W_hc) # Modified
            # H = O * tf.tanh(C)+ C_tilde #To Check
            outputs.append(H)
        return tf.stack(outputs, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hiddens": self.num_hiddens,
            "sigma": self.sigma,
            "kernel_regularizer": self.kernel_regularizer
        })
        return config
