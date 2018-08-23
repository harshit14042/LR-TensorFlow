import tensorflow as tf
import pickle
import os
print("Initialization");
lr=0.5
epochs=500
batch_size=10
step=1

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

W=tf.Variable(tf.random_normal([len(X_Data[0]),1]),name='W');
b=tf.Variable(tf.random_normal([1]),name='b');

pred=tf.nn.sigmoid(tf.matmul(x,W)+b);
delta=pow(10,-9);

cost=tf.reduce_mean(-y*(tf.log(pred+delta))-(1-y)*tf.log(1-pred+delta));
#cost=tf.reduce_mean(-tf.reduce_mean(y*tf.log(pred+delta)));

optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost);

init = tf.global_variables_initializer();

path='./model/';
saver=tf.train.Saver();

print("Training Starts");

z=[n.name for n in tf.get_default_graph().as_graph_def().node]
print(z);

with tf.Session() as sess:
    sess.run(init);

    for e in range(epochs):

        avg_cost=0.0;
        for i in range(0,len(X_train),batch_size):
            batch_x=X_train[i:min(i+batch_size,len(X_train))];
            batch_y=Y_train[i:min(i+batch_size,len(Y_train))];

            p,_,c=sess.run([pred,optimizer,cost],feed_dict={x:batch_x,y:batch_y});

            #print(1 in p);

            avg_cost+=c/batch_size;


        if ((e+1)%step==0):
            print("Epoch: "+str(e));
            print("Loss: "+str((avg_cost*batch_size)/len(X_train)));

        if((e+1)%10==0):
            saver.save(sess,os.path.join(path,'model'),global_step=e+1);


