# Reconocimiento de Emociones con LBPH usando OpenCV

Este proyecto consiste utilizar la libreria OpenCV para implementar un sistema de reconocimiento de emociones basado en el algoritmo LBPH (Local Binary Patterns Histogram). Las emociones que se pretende identificar con el modelo son Tristeza, Felicidad y sorpresa. 

## Creación del Modelo

En esta primer parte del proyecto vamos a crear el modelo usando el algortimo LBPH

```python
# Importación de bibliotecas necesarias
import cv2 as cv
import numpy as np
import os

# Definimos la ruta del conjunto de datos que contiene las imágenes de las emociones
dataSet = 'dataset/emociones' #El dataset 'emociones' esta cargado en este repositorio
# Listamos los directorios dentro de la carpeta del conjunto de datos, cada uno correspondiente a una emoción
faces = os.listdir(dataSet)
#Imprime los nombres de los subdirectorios (emociones)
print(faces)  

# Inicializamos las listas para etiquetas y datos de las caras
labels = []
facesData = []
# Inicializamos el contador de etiquetas
label = 0

# Iteramos sobre cada emoción en la lista de emociones
for face in faces:
    # Construimos la ruta completa hacia el directorio de la emoción actual
    facePath = dataSet + '/' + face
    # Iteramos sobre cada imagen dentro del directorio de la emoción actual
    for faceName in os.listdir(facePath):
        # Añadimos la etiqueta actual a la lista de etiquetas
        labels.append(label)
        # Añadimos los datos de la imagen de la cara a la lista de datos de caras 
        facesData.append(cv.imread(facePath + '/' + faceName, 0))  # 0 indica que se carga la imagen en escala de grises
    # Incrementamos la etiqueta para la siguiente emoción
    label += 1  

# Creamos el reconocedor de caras utilizando el método LBPH (Local Binary Patterns Histograms)
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
# Entrenamos el reconocedor de caras con los datos de las caras y las etiquetas correspondientes
faceRecognizer.train(facesData, np.array(labels))


# Guardar el modelo entrenado en un archivo XML

faceRecognizer.write('EmocionesLBPHF.xml')
```


## Implementación del Modelo

En la segunda Etapa del proyecto se evalua el Modelo entrenado en la etapa anterior

```python

# Importamos la biblioteca OpenCV
import cv2 as cv
import os 

# Creamos una instancia del reconocedor de caras usando el método LBPH
faceRecognizer = cv.face.LBPHFaceRecognizer_create()

# Cargamos el modelo entrenado previamente desde un archivo XML
faceRecognizer.read('EmocionesLBPHF.xml')

# Inicializamos la captura de video desde la cámara (índice 0)
cap = cv.VideoCapture(0)

# Cargamos el clasificador Haar Cascade para la detección de rostros
rostro = cv.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

# Bucle infinito para procesar el video en tiempo real
while True:
    # Leemos un frame de la cámara
    ret, frame = cap.read()
    
    # Si la lectura del frame falla, salimos del bucle
    if ret == False: break
    
    # Convertimos el frame a escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Creamos una copia de la imagen en escala de grises
    cpGray = gray.copy()
    
    # Detectamos rostros en la imagen en escala de grises
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    
    # Iteramos sobre cada rostro detectado
    for(x, y, w, h) in rostros:
        # Extraemos la región de interés (el rostro) y la redimensionamos a 100x100 píxeles
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100, 100), interpolation=cv.INTER_CUBIC)
        
        # Usamos el modelo de reconocimiento facial para predecir la emoción
        result = faceRecognizer.predict(frame2)
        
        # Escribimos el resultado de la predicción en el frame
        cv.putText(frame, '{}'.format(result), (x, y-20), 1, 3.3, (255, 255, 0), 1, cv.LINE_AA)
        
        # Si la confianza de la predicción es mayor a 70, consideramos que se ha reconocido una emoción
        if result[1] > 70:
            # Escribimos el nombre de la emoción en el frame
            cv.putText(frame, '{}'.format(faces[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            # Dibujamos un rectángulo verde alrededor del rostro reconocido
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # Si la confianza es baja, consideramos que el rostro es desconocido
            cv.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            # Dibujamos un rectángulo rojo alrededor del rostro desconocido
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Mostramos el frame procesado en una ventana
    cv.imshow('frame', frame)
    
    # Esperamos 1 ms a que el usuario presione una tecla; si es la tecla Esc (código 27), salimos del bucle
    k = cv.waitKey(1)
    if k == 27:
        break

# Liberamos la captura de video y cerramos todas las ventanas de OpenCV
cap.release()
cv.destroyAllWindows()
```
