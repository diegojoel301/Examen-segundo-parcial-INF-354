import numpy as np
from sklearn import datasets

# Motivo del uso de sigmoide para el dataset iris:
# Problema de clasificacion binaria: Originalmente, la funcion sigmoide se usaba ampliamente para problemas de clasificación binaria.
# Aunque el conjunto de datos de iris tiene tres clases,
# la funcion sigmoide podria usarse para una version simplificada del problema donde solo se distinguen dos clases
# (por ejemplo, una especie de iris frente a las demas)

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
data = iris.data
y_trues = iris.target
class_labels = iris.target_names

# Definición de la función sigmoide y su derivada
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_del_sigmoide(x):
    fx = sigmoide(x)
    return fx * (1 - fx)

# Definición de la función de pérdida Mean Squared Error (MSE)

# La funcion de perdida de error cuadratico medio,
# que mide el promedio de los errores al cuadrado entre las
# etiquetas verdaderas y las predicciones.

def mse_perdida(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Definición de la clase RedNeuronal
class RedNeuronal:

    def __init__(self):
        # Inicialización de pesos y umbrales para cada capa
        self.p1 = np.random.normal()
        self.p2 = np.random.normal()
        self.p3 = np.random.normal()
        self.p4 = np.random.normal()
        self.b1 = np.random.normal()

        self.p5 = np.random.normal()
        self.p6 = np.random.normal()
        self.p7 = np.random.normal()
        self.p8 = np.random.normal()
        self.b2 = np.random.normal()

        self.p9 = np.random.normal()
        self.p10 = np.random.normal()
        self.p11 = np.random.normal()
        self.b3 = np.random.normal()

    def retroalimentacion(self, x):
        # Feedforward a través de la red neuronal
        # Primera capa oculta
        sum_neurona1 = self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.b1
        neurona1 = sigmoide(sum_neurona1)

        sum_neurona2 = self.p5 * x[0] + self.p6 * x[1] + self.p7 * x[2] + self.p8 * x[3] + self.b2
        neurona2 = sigmoide(sum_neurona2)

        # Segunda capa oculta
        sum_neurona3 = self.p9 * neurona1 + self.p10 * neurona2 + self.b3
        neurona3 = sigmoide(sum_neurona3)

        return neurona3

    # Haremos uso del algoritmo de retropropagacion y metodo de descenso de gradiente
    def train(self, datos, y_trues):
        # Parámetros de entrenamiento
        tasa_aprendizaje = 0.1
        epocas = 1000

        for epoca in range(epocas):
            for x, y_true in zip(datos, y_trues):
                # Feedforward
                sum_neurona1 = self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.b1
                neurona1 = sigmoide(sum_neurona1)

                sum_neurona2 = self.p5 * x[0] + self.p6 * x[1] + self.p7 * x[2] + self.p8 * x[3] + self.b2
                neurona2 = sigmoide(sum_neurona2)

                sum_neurona3 = self.p9 * neurona1 + self.p10 * neurona2 + self.b3
                neurona3 = sigmoide(sum_neurona3)

                # BackPropagation
                # Calcula la derivada parcial del error respecto a la prediccion
                # dL/dy^​= −2(y− y^​)
                d_L_d_ypred = -2 * (y_true - neurona3)

                # Derivadas para la segunda capa oculta

                # Se necesita calcular la derivada de la funcion de perdida con respecto a su respectivo
                # peso y sesgo, por lo que se calcula dL/dwi = dL/dy^​ * dy^​/dwi

                d_ypred_d_p9 = neurona1 * derivada_del_sigmoide(sum_neurona3)
                d_ypred_d_p10 = neurona2 * derivada_del_sigmoide(sum_neurona3)
                d_ypred_d_b3 = derivada_del_sigmoide(sum_neurona3)

                # Derivadas para la primera capa oculta
                d_ypred_d_neurona1 = self.p9 * derivada_del_sigmoide(sum_neurona3)
                d_ypred_d_neurona2 = self.p10 * derivada_del_sigmoide(sum_neurona3)

                # La derivada de la prediccion y^ con respecto al peso wi, depende de la activacion 
                # de la neurona y su derivada del sigmoide

                d_neurona1_d_p1 = x[0] * derivada_del_sigmoide(sum_neurona1)
                d_neurona1_d_p2 = x[1] * derivada_del_sigmoide(sum_neurona1)
                d_neurona1_d_p3 = x[2] * derivada_del_sigmoide(sum_neurona1)
                d_neurona1_d_p4 = x[3] * derivada_del_sigmoide(sum_neurona1)
                d_neurona1_d_b1 = derivada_del_sigmoide(sum_neurona1)

                d_neurona2_d_p5 = x[0] * derivada_del_sigmoide(sum_neurona2)
                d_neurona2_d_p6 = x[1] * derivada_del_sigmoide(sum_neurona2)
                d_neurona2_d_p7 = x[2] * derivada_del_sigmoide(sum_neurona2)
                d_neurona2_d_p8 = x[3] * derivada_del_sigmoide(sum_neurona2)
                d_neurona2_d_b2 = derivada_del_sigmoide(sum_neurona2)


                # Actualizar

                # Una vez que se han calculado los gradientes, los pesos y sesgos se actualizan
                # en la dirección que minimiza la perdida.
                # La regla de actualización para un peso es
                # w = w - tasa_aprendizaje * dL/dw
                # La regla de actualización para el sesgo es
                # b = b - tasa_aprendizaje * dL/db

                self.p9 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_p9
                self.p10 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_p10
                self.b3 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_b3

                self.p1 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p1
                self.p2 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p2
                self.p3 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p3
                self.p4 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p4
                self.b1 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_b1

                self.p5 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p5
                self.p6 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p6
                self.p7 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p7
                self.p8 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p8
                self.b2 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_b2

            # Imprimir la pérdida cada 10 épocas
            if epoca % 10 == 0:
                y_preds = np.apply_along_axis(self.retroalimentacion, 1, data)
                perdida = mse_perdida(y_trues, y_preds)
                print("Época %d Pérdida: %.3f" % (epoca, perdida))

    def predict(self, x):
        # Normalizar el dato de entrada
        x_normalized = (x - np.mean(data, axis=0)) / np.std(data, axis=0)

        # Calcular la salida de la red neuronal
        output = self.retroalimentacion(x_normalized)

        # Convertir las salidas a probabilidades (0 o 1)
        binary_predictions = (output > 0.5).astype(int)

        # Traducir las probabilidades binarias a clases usando las etiquetas de clase del conjunto de datos Iris
        predicted_class = binary_predictions.argmax()

        return predicted_class

# Crear y entrenar la red neuronal
mired = RedNeuronal()
mired.train(data, y_trues)

# Realizar una predicción
data1 = np.array([6.1, 2.8, 4, 1.3])
predicted_class = mired.predict(data1)
print("Predicción:", class_labels[predicted_class])
