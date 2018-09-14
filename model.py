import tensorflow as tf
import numpy as np
import pickle


#Loading in picled data from Leete
training = pickle.load(open('training.dat', 'rb'))
testing = pickle.load(open('testing.dat', 'rb'))
val = pickle.load(open('validation.dat', 'rb'))

#Preparing Loaded data
X_train = training['X']
Y_train = training['Y']
X_val = val['X']
Y_val = val['Y']
X_test = testing['X']



#defining layer paramerters
inputs = 15
layer_1_neurons = 15
layer_2_neurons = 10
layer_3_neurons = 5
output_neurons = 1

learning_rate = 0.001
epochs = 500
batch_size = 128

#Creating place holders for inputs and outputs
X = tf.placeholder(dtype=tf.float32, shape=[None, inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_neurons])

#defining layer attributes
h_1_layer_weights = tf.Variable(tf.random_normal([inputs, layer_1_neurons]))
h_1_layer_bias = tf.Variable(tf.random_normal([layer_1_neurons]))

h_2_layer_weights = tf.Variable(tf.random_normal([layer_1_neurons, layer_2_neurons]))
h_2_layer_bias = tf.Variable(tf.random_normal([layer_2_neurons]))

h_3_layer_weights = tf.Variable(tf.random_normal([layer_2_neurons, layer_3_neurons]))
h_3_layer_bias = tf.Variable(tf.random_normal([layer_3_neurons]))

output_weights = tf.Variable(tf.random_normal([layer_3_neurons, output_neurons]))
output_bias = tf.Variable(tf.random_normal([output_neurons]))

#defining layer calculators and flow
#REMEMBER Weights
l1_calc = tf.nn.relu(tf.add(tf.matmul(X, h_1_layer_weights), h_1_layer_bias))
l2_calc = tf.nn.relu(tf.add(tf.matmul(l1_calc, h_2_layer_weights), h_2_layer_bias))	
l3_calc = tf.nn.relu(tf.add(tf.matmul(l2_calc, h_3_layer_weights), h_3_layer_bias))
output_calc = tf.add(tf.matmul(l3_calc, output_weights), output_bias)

#defining training, cost, and accuracy calculations
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_calc, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
predicted_class = tf.equal(tf.round(tf.nn.sigmoid(output_calc)), tf.round(Y))
accuracy = tf.reduce_mean(tf.cast(predicted_class, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #initalizes the varibles in prep for training


    for e in range(0, epochs):
        epoch_loss = 0 
        shuffle_set = np.random.permutation(np.arange(len(Y_train))) #randomly shuffles data for each epoch to combat overfitting
        X_train = X_train[shuffle_set]
        Y_train = Y_train[shuffle_set]



        for i in range(0, len(Y_train) // batch_size):
            print("Batch: {0}".format(str(i) + "/" + str(len(Y_train) // batch_size)), end="\r")
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = Y_train[start:start + batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y}) #runs the batch in the optimizer and gets the cost
            epoch_loss += c #compounds cost for epoch

        if e % 1 == 0:
            testing_acc = accuracy.eval({X: X_val, Y: Y_val}) #evaluates accuracy of model on unseen data
            training_acc = accuracy.eval({X: X_train, Y: Y_train}) #evaluates accracy of model on seen data

            print('[+] -----   Epoch: ', e, 'Accuracy on Unseen Data:', testing_acc, 'Accuracy on TRAINING data: ', training_acc, 'Training Loss: ', epoch_loss)