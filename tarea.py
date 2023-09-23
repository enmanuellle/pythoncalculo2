import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit, QMessageBox, QComboBox
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
import datetime
from PyQt5.QtCore import QTimer, QTime
import numpy as np
import gausiana 




class MiAplicacion(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('diseno.ui', self)  # Carga el diseño .ui creado en Qt Designer
        self.initUI()

    def initUI(self):
        # Logica de la aplicación
        #conección al diseño
        


        self.text_edit1 = self.findChild(QTextEdit, 'gausiana1')
        self.text_edit2 = self.findChild(QTextEdit, 'gaussiana2')
        self.text_edit3 = self.findChild(QTextEdit, 'potencia1')
        self.text_edit4 = self.findChild(QTextEdit, 'potencia2')
        self.text_edit5 = self.findChild(QTextEdit, 'potencia3')
        self.text_edit6 = self.findChild(QTextEdit, 'ps1')
        self.text_edit7 = self.findChild(QTextEdit, 'ps2')
        self.text_edit8 = self.findChild(QTextEdit, 'ps3')

        self.asimetrica.clicked.connect(self.gausiana.gausianaclase.eliminar_gaussiano)
        self.ainve.clicked.connect(self.gausiana.gausianaclase.eliminar_gaussiano)
        self.agaus.clicked.connect(self.gausiana.gausianaclase.eliminar_gaussiano)





    def is_symmetric(matrix):
        # Comprueba si la matriz es simétrica comparando con su traspuesta.
        return np.allclose(matrix, matrix.T)

    def symmetric_power_iteration(self, A, num_iterations, tolerance):
        """
        Encuentra el eigenvalor dominante y eigenvector correspondiente de una matriz simétrica utilizando el Método de Potencia Simétrico.

        :param A: La matriz simétrica de entrada.
        :param num_iterations: Número máximo de iteraciones.
        :param tolerance: Tolerancia para la convergencia.
        :return: Eigenvalor dominante y eigenvector correspondiente.
        """
        if not is_symmetric(A):
            raise ValueError("La matriz de entrada no es simétrica. Este algoritmo solo funciona con matrices simétricas.")

        n = A.shape[0]

        # Paso 1: Inicializar un vector aleatorio como aproximación inicial al eigenvector.
        v = np.random.rand(n)

        for i in range(num_iterations):
            # Paso 2: Calcula el producto A * v.
            Av = np.dot(A, v)
            
            # Paso 3: Normaliza el eigenvector.
            v_next = Av / np.linalg.norm(Av)
            
            # Paso 4: Calcula el eigenvalor estimado.
            eigenvalue = np.dot(v_next, np.dot(A, v_next))
            
            # Paso 5: Comprueba la convergencia.
            if np.linalg.norm(v_next - v) < tolerance:
                break
            
            # Actualiza el eigenvector para la próxima iteración.
            v = v_next

            

        return eigenvalue, v

        # Llama a la función para encontrar el eigenvalor dominante y eigenvector correspondiente
        eigenvalue, eigenvector = symmetric_power_iteration(A, num_iterations, tolerance)

        # Imprime los resultados
        print("Eigenvalor estimado:", eigenvalue)
        print("Eigenvector estimado:", eigenvector)

    def eliminar_gaussiano(self, A, b):

        A= A.toPlainText()
        b= b.toPlainText()
        n = len(A)

        # Eliminación hacia adelante
        for i in range(n):
            pivot = A[i][i]
            for j in range(i + 1, n):
                factor = A[j][i] / pivot
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
                b[j] -= factor * b[i]

        # Sustitución hacia atrás
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = b[i] / A[i][i]
            for j in range(i - 1, -1, -1):
                b[j] -= A[j][i] * x[i]

        return x

        solucion = (A, b)
        print("La solución del sistema es:", solucion)


        import numpy as np
        from scipy.linalg import lu_solve, lu_factor

    def inverse_power_iteration(self, A, num_iterations, tolerance):
        """
        Encuentra el eigenvalor más pequeño y el eigenvector correspondiente de una matriz utilizando el Método de Potencia Inverso.

        :param A: La matriz de entrada.
        :param num_iterations: Número máximo de iteraciones.
        :param tolerance: Tolerancia para la convergencia.
        :return: Eigenvalor más pequeño y eigenvector correspondiente.
        """
        n = A.shape[0]

        # Paso 1: Inicializar un vector aleatorio como aproximación inicial al eigenvector.
        v = np.random.rand(n)

        # Factoriza la matriz A - sigma * I utilizando la factorización LU
        lu, piv = lu_factor(A)

        for i in range(num_iterations):
            # Paso 2: Resuelve el sistema de ecuaciones lineales (A - sigma * I) * x = v utilizando la factorización LU
            x = lu_solve((lu, piv), v)
            
            # Paso 3: Normaliza el eigenvector.
            v_next = x / np.linalg.norm(x)
            
            # Paso 4: Calcula el eigenvalor estimado.
            eigenvalue = 1 / np.dot(v_next, x)
            
            # Paso 5: Comprueba la convergencia.
            if np.linalg.norm(v_next - v) < tolerance:
                break
            
            # Actualiza el eigenvector para la próxima iteración.
            v = v_next

        return eigenvalue, v



        # Llama a la función para encontrar el eigenvalor más pequeño y el eigenvector correspondiente
        eigenvalue, eigenvector = inverse_power_iteration(A, num_iterations, tolerance)

        # Imprime los resultados
        print("Eigenvalor más pequeño estimado:", eigenvalue)
        print("Eigenvector correspondiente:", eigenvector)








        


    #función para salir de la app 
    def salir_app(self):
         sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MiAplicacion()
    window.show()
    sys.exit(app.exec_())