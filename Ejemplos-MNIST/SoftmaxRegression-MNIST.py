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
VARIABLES:

Tensorflow maneja el sesgo y los pesos de una manera eficiente mediante "Variables".
Estas variables viven en el grafo de cálculos y pueden ser usadas e incluso modificadas
por el cálculo.
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
'''
Inicializamos W y b como tensores nulos (llenos de ceros).
W: será una matriz de 784x10 porque hay 784 características de entrada y sólo diez salidas.
b: será un vector de dimensión 10 porque sólo hay 10 clases.
'''

sess.run(tf.initialize_all_variables())

'''
PREDICCION Y FUNCION DE COSTO
'''
#Predicción
y = tf.matmul(x,W) + b
#Función de costo
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

'''
ENTRENAMIENTO:
Debido a que en este punto tenemos todo nuestro modelo definido, Tensorflow conoce todo el grafo
de cálculos, por lo que es posible utilizar diferenciación automática para encontrar los gradientes
de la función de costo. En este punto Tensorflow dispone de muchas funciones para realizar el cálculo,
pero utilizamos steepest gradient descent, con 0.5 de constante de aprendizaje.
'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
'''
La operación train_step devuelve los pesos actualizados, por lo que podemos conseguir el descenso
de gradiente aplicando train_step de manera iterativa.
'''
for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  '''
  Esto permite cargar 100 imágenes de entrenamiento en cada iteración.
  Usando feed_dict reemplazamos a los tensores x e y_ con los ejemplos de entrenamiento.
  Se puede reemplazar a cualquier tensor en un grafo utilizando feed_dict
  '''

'''
EVUALUANDO EL MODELO:
tf.argmax: devuelve el máximo valor en un tensor a lo largo del mismo eje.
  tf.argmax(y,1): serían los valores que predice nuestro modelo
  tf.argmax(y_,1): son los valores reales que debería predecir nuestro modelo.
tf.equal: compara los resultados de nuestras predicciones y las predicciones reales. Devuelve un vector de True y False.
'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))