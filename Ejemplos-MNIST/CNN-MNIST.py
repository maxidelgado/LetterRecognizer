# coding=utf-8
import tensorflow as tf

#Importa el conjunto de entrenamiento MNIST en formato NumPy.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Inicia una sesion de Tensorflow que llama a un backend en C++ muy eficiente.
sess = tf.InteractiveSession()

'''
Tensorflow describe un grafo de operaciones que van a ser realizadas fuera de Python.
Este enfoque se utiliza tambien en Torch o Theano.
'''


'''
PLACEHOLDERS:

Las imagenes de entrada x consistiran en un tensor 2D de numeros de punto flotante.
Le damos la forma [None, 784], donde 784 corresponde a 28x28 píxels de las imagenes de MNIST
y None indice que la dimension del lote puede ser de cualquier tamaño.

La salida y_ es de forma [None, 10] ya que hay 10 posibles valores de salida (del 0 al 9)
'''
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

'''
COSTRUIR LA RED CONVOLUCIONAL MULTICAPA:
Para crear este modelo es necesario inicializar los pesos y los sesgos con valores random y pequeños, para evitar
los gradientes cero y las neuronas muertas.
'''
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
El operador de convolución tiene el efecto de filtrar la imagen de entrada con un núcleo previamente entrenado.
Esto transforma los datos de tal manera que ciertas características (determinadas por la forma del núcleo) se vuelven
más dominantes en la imagen de salida al tener estas un valor numérico más alto asignados a los pixeles que las
representan.

La operación de max-pooling encuentra el valor máximo entre una ventana de muestra y pasa este valor como resumen de
características sobre esa área. Como resultado, el tamaño de los datos se reduce por un factor igual al tamaño de la
ventana de muestra sobre la cual se opera.
'''

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

'''
PRIMERA CAPA DE CONVOLUCION:
Consiste en una convolucion seguida de un max-pooling.
La convolucion calculará 32 características de cada bloque de 5x5. Está determinado por un tensor  [5, 5, 1, 32],
donde los dos primeros números son las dimensiones del tamaño de bloque, el tercer número es el número de canales de
entrada y el último son los canales de salida.
También tendremos un vector de sesgos con un componente para cada canal de salida.
'''

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

'''
Hacemos un reshape de x, transformándolo en un tensor 4D.
El segundo y tercer número son las dimensiones de la imagen.
El cuarto número es el número de canales de color de la imagen.
'''

x_image = tf.reshape(x, [-1,28,28,1])

'''
Convolucionamos x_image
'''

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
Agregamos la segunda capa de convolucion, que extrae 64 características en lugar de 32
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
En este punto la imagen ya fue reducida a un tamaño de 7x7.
Agregamos una capa completamente conectada con 1024 neuronas para procesar la imagen entera.
Hacemos un reshape del tensor proveniente de la capa de pooling y lo transformamos en un lote de vectores.
'''

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
Agregamos un mecanismo de control de overfitting llamado "Dropout"
'''

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
Finalmente agregamos una capa final de lectura que será como una regresión Softmax
'''

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''
Finalmente se entrena y evualua el modelo.
Para ello se sustituye al optimizador de descenso de gradiente por otro más sofisticado llamado ADAM
'''

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))