import tensorflow as tf
import pickle
import os
import numpy as np
print("Initialization");
lr=0.01
epochs=100
batch_size=300
step=1

model_path = "./model/"
model_name = "model-500"
model_name_meta = "model-500.meta"

X_Data=pickle.load(open('X','rb'));
Y_Data=pickle.load(open('Y','rb'));
n=len(X_Data);
split=int(0.8*n);
X_train=X_Data[0:split];
Y_train=Y_Data[0:split];
X_test=X_Data[split:n]
Y_test=Y_Data[split:n];

print("input");
x=tf.placeholder(tf.float32,[None,len(X_Data[0])]);
y=tf.placeholder(tf.float32,[None,1]);
sess=tf.Session();
path='./model/';
saver = tf.train.import_meta_graph('./model/model-500.meta')
saver.restore(sess,os.path.join(path,'model-500'))
graph = tf.get_default_graph()

W = graph.get_tensor_by_name("W:0")
b = graph.get_tensor_by_name("b:0")
pred=tf.nn.sigmoid(tf.matmul(x,W)+b);
delta=pow(10,-9);

cost=tf.reduce_mean(-y*(tf.log(pred+delta))-(1-y)*tf.log(1-pred+delta));
#cost=tf.reduce_mean(-tf.reduce_mean(y*tf.log(pred+delta)));
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost);

path='./model/';
saver=tf.train.Saver();
cord=tf.train.Coordinator();

print("Testing Starts");


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(os.path.join(model_path, model_name_meta))

    new_saver.restore(sess, os.path.join(model_path, model_name))

    graph = tf.get_default_graph()
    avg=0;
    t=0;
    tot=0
    for i in range(0,len(X_test),batch_size):
        tot=tot+1
        out = sess.run(pred, feed_dict={x: X_test[i:min(i+batch_size,len(X_test))]});
        l=0;
        z=abs(np.ndarray.round(out) - Y_test[i:min(i+batch_size,len(X_test))]);
        for j in range(len(out)):
            l+=z[j]

        avg+=(l);
        if(0 in out and 1 in out):
            t=t+1
            print(np.ndarray.round(out));
            print(Y_test[i:min(i+batch_size,len(X_test))])
            print((l / len(out)));

    avg=avg/len(X_test);
    print(t)
    print(tot)
    print(1-avg)