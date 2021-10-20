import cv2
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt

# Crea estructura de datos de video a leer
video_camara = cv2.VideoCapture(0)
video_camara.set(3, 1280)
video_camara.set(4, 960)

# inicializa media pipe
mp_pose = mp.solutions.pose

# Configura la posicion para imagenes
pose_imagen = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Configura la posicion para videos
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,min_tracking_confidence=0.7)

# inicia la libreria media pipe
mp_drawing = mp.solutions.drawing_utils

def detectar_pose(image, pose, dibujo=False, display=False):
    salida_imagen = image.copy()
    # de BGR a RGB
    imagenRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Realizar la detección de posicion.
    results = pose.process(imagenRGB)
    #Compruebe si se detectan puntos de referencia y se especifica que se van a dibujar.
    if results.pose_landmarks and dibujo:
        # puntos de referencia en la imagen de salida
        mp_drawing.draw_landmarks(image=salida_imagen, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),thickness=3,circle_radius=3),connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237),thickness=2, circle_radius=2))

    if display:
        plt.figure(figsize=[22, 22])
        plt.subplot(121);plt.imshow(image[:, :, ::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(salida_imagen[:, :, ::-1]);plt.title("Output Image");plt.axis('off');
    else:
        return salida_imagen, results

# Deteccion de puntos de la posicion en la imagen
IMG_PATH = 'sample.jpg'
image = cv2.imread(IMG_PATH)
detectar_pose(image, pose_imagen, dibujo=True, display=True)

def detectar_manos(image, results, dibujo=False, display=False):

    height, width, _ = image.shape

    # Estado de las manos
    salida_imagen = image.copy()

    #  coordenadas X Y del punto de referencia de la muñeca izquierda
    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

    # coordenadas X Y del punto de referencia de la muñeca derecha
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

    # Distancia parabolica entre muñeca izquierda y derecha
    distancia = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],left_wrist_landmark[1] - right_wrist_landmark[1]))

    # Distancia entre las muñecas para verificar si las manos estan unidas
    if distancia < 130:

        hand_status = 'Manos unidas'
        color = (0, 255, 0)
    else:
        hand_status = 'Manos sin unir'
        color = (0, 0, 255)

    if dibujo:
        # Estados de las manos
        cv2.putText(salida_imagen, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(salida_imagen, f'Distancia: {distancia}', (10, 70),cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(salida_imagen[:, :, ::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Verifica si las manos estan unidas
        return salida_imagen, hand_status

def izquierda_derecha(image, results, dibujo=False, display=False):
    # Declara una variable para almacenar la posición horizontal (izquierda, centro, derecha) de la persona.
    posicion_x = None
    # Obtener la altura y el ancho de la imagen.the image.
    height, width, _ = image.shape
    # Crea una copia de la imagen de entrada para escribir la posición horizontal.
    salida_imagen = image.copy()
    # la coordenada x del  del hombro izquierdo.
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
    # la coordenada x del  del hombro derecho.
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    # Verifica si la persona está a la izquierda, cuando ambos puntos de referencia de hombros se coordinan en x
    # son menores o iguales a la coordenada x del centro de la imagen
    if (right_x <= width // 2 and left_x <= width // 2):
        # Establecer la posición de la persona a la izquierda.
        posicion_x = 'left'
    # Verifique si la persona está a la derecha, es decir, cuando ambos puntos de referencia de hombros se coordinan en x
    # son mayores o iguales a la coordenada x del centro de la imagen.
    elif (right_x >= width // 2 and left_x >= width // 2):
        # Establece la posición de la persona en la derecha.
        posicion_x = 'Right'
    # Verifique si la persona está en el centro, es decir, cuando la coordenada x del punto de referencia del hombro derecho es mayor o igual que
    # el hombro izquierdo x-corrdinate es menor o igual que la coordenada x del centro de la imagen.
    elif (right_x >= width // 2 and left_x <= width // 2):
        # Establecer la posición de la persona en el centro.
        posicion_x = 'Centro'
    # Comprueba si se especifica la posición horizontal de la persona y una línea en el centro de la imagen para dibujar.
    if dibujo:
        # Escribe la posición horizontal de la persona en la imagen.
        cv2.putText(salida_imagen, posicion_x, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        # Dibuja una línea en el centro de la imagen.
        cv2.line(salida_imagen, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
    # Compruebe si la imagen de salida está especificada para mostrarse.
    if display:
        # Mostrar la imagen de salida.
        plt.figure(figsize=[10, 10])
        plt.imshow(salida_imagen[:, :, ::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Retorna la imagen de salida y la posición horizontal de la persona.
        return salida_imagen, posicion_x

def salto(image, results, MID_Y=250, dibujo=False, display=False):
    height, width, _ = image.shape
    # Etiqueta de posicion
    salida_imagen = image.copy()
    # Punto de referencia de hombro izquierdo
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    # Punto de referencia de hombro derecho
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    # Punto medio de los hombros
    actual_mid_y = abs(right_y + left_y) // 2
    # Calcular los limites superiores o inferiores del umbral
    lower_bound = MID_Y - 15
    upper_bound = MID_Y + 100

    if (actual_mid_y < lower_bound):
        posture = 'Saltando'
    elif (actual_mid_y > upper_bound):
        posture = 'Agachado'
    else:
        posture = 'Estatico'
    # Especificar una posicion
    if dibujo:
        # Escribir posicion de la persona
        cv2.putText(salida_imagen, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        # Dibujamos la linea de coordenadas
        cv2.line(salida_imagen, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)
    if display:
        # Imagen de salida
        plt.figure(figsize=[10, 10])
        plt.imshow(salida_imagen[:, :, ::-1]);plt.title("Output Image");plt.axis('off');
    else:
        # Devolver la imagen de salida y la posicion indicando si esta de pie saltando o agachado
        return salida_imagen, posture

# Crear ventana
cv2.namedWindow('Subway Surfers Detector de posicion', cv2.WINDOW_NORMAL)
# tiempo del frame anterior
time1 = 0
# Almacenamos el estado del juego
juego = False
# Variable inicial para la posicion horizontal de la persona
# En el centro 1
# En la izquierda es 0
# En la derecha es 2
x_pos_index = 1
# Variable para la posicion actual
# La persona inicia de pie 1
# si se agacha es 0
# si salta es 2
y_pos_index = 1
# Variable para almacenar coordenadas del punto inicial en el medio de los hombros de la persona
MID_Y = None
# inicia un contador para almacenar el numero de frames con las manos unidas
contador = 0
# inicia el numero de frames consecutivos en los que queremos comprabar las manos de la persona para iniciar el juego
num_frame = 10
# Lo hace siempre que la camara se en cuentra abierta correctamente

while video_camara.isOpened():
    # Read a frame.
    ok, frame = video_camara.read()
    # Compruebe si el fotograma no se lee correctamente y luego continúe con la siguiente iteración para leer el siguiente fotograma.
    if not ok:
        continue
    # Voltear el marco horizontalmente para una visualización natural.
    frame = cv2.flip(frame, 1)
    # Obtener la altura y el ancho del marco del video de la cámara web
    frame_height, frame_width, _ = frame.shape
    # Detección de posicion en el marco.
    frame, results = detectar_pose(frame, pose_video, dibujo=juego)
    # Comprueba si se detectan los puntos de referencia de posicion en el marco.
    if results.pose_landmarks:
        # Comprueba si el juego ha comenzado
        if juego:
            # Comandos para controlar los movimientos horizontales del personaje.
            # Obtener la posición horizontal de la persona en el marco.
            frame, posicion_x = izquierda_derecha(frame, results, dibujo=True)
            # Compruebe si la persona se ha movido a la izquierda desde el centro o al centro desde la derecha.
            if (posicion_x == 'left' and x_pos_index != 0) or (posicion_x == 'Centro' and x_pos_index == 2):
                # Presione la tecla de flecha izquierda.
                pyautogui.press('left')
                # Actualizar el índice de posición horizontal del personaje.
                x_pos_index -= 1
                # Compruebe si la persona se ha movido a la derecha desde el centro o al centro desde la izquierda.
            elif (posicion_x == 'Right' and x_pos_index != 2) or (posicion_x == 'Centro' and x_pos_index == 0):
                # Presione la tecla de flecha derecha.
                pyautogui.press('Right')
                # Actualizar el índice de posición horizontal del personaje.
                x_pos_index += 1
        # si el juego no se ha iniciado
        else:
            # Escribe el texto que representa la forma de comenzar el juego en el marco.
            cv2.putText(frame, 'Une las manos para iniciar el juego.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255), 3)
        # Comando para iniciar o reanudar el juego.
        # Comprueba si las manos izquierda y derecha están unidas.# Comando para iniciar o reanudar el juego.
        if detectar_manos(frame, results)[1] == 'Manos unidas':
            # Incrementar el recuento de fotogramas consecutivos con condición +ve.
            contador += 1
            # Comprueba si el contador es igual al número requerido de fotogramas consecutivos.
            if contador == num_frame:
                #Funcion para iniciar el juego por primera vez
                # Compruba si el juego aun no se ejecuta
                if not (juego):
                    # Actualiza el valor de la variable que almacena el estado del juego
                    juego = True
                    # Retoma la coordenada del hombro izquierdo al punto de referencia
                    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
                    # Retoma la coordenada del hombro derecho al punto de referencia
                    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
                    # Calcula posicion de ambos hombros de la persona en el punto medio
                    MID_Y = abs(right_y + left_y) // 2
                    # Para comenzar el juego da click
                    pyautogui.click(x=1300, y=800, button='Left')
                # Funcion para reanudar el juego luego de perder
                # si no inicia el juego
                else:
                    # Presiona la tecla espacio
                    pyautogui.press('Space')
                # carga contador 0
                contador = 0
        else:
            # carga contador 0
            contador = 0
        # Comando para controlar los movimientos
        # Funcion para Comprobar las coordenadas en Y inicial en el punto medio de los hombros de la persona
        if MID_Y:
            # Funcion para verificar la posicion en el frame (saltar, agacharse, pararse)
            frame, posture = salto(frame, results, MID_Y, dibujo=True)
            # comprueba si la persona esta en la posicion saltando
            if posture == 'Saltando' and y_pos_index == 1:
                # Presiona la tecla de flecha hacia arriba
                pyautogui.press('Up')
                # Actualiza la posicion vertical inicial
                y_pos_index += 1
                # Comprueba si la persona esta agachada.
            elif posture == 'Agachado' and y_pos_index == 1:
                # Presiona la tecla de flecha hacia abajo
                pyautogui.press('Down')
                # Actualiza la posicion vertical de la persona
                y_pos_index -= 1
                # Comprueba si la persona se encuentra de pie
            elif posture == 'Estatico' and y_pos_index != 1:
                # Actualiza la posicion vertical inicial
                y_pos_index = 1
    # Si no detecta la posicion en el punto de referencia
    else:
        # carga contador 0
        contador = 0
    # Establece la hora de este fotograma en la hora actual.
    time2 = time()
    # Comprueba si la diferencia entre el tiempo de fotograma anterior y este mayor que 0 para evitar la división por cero.
    if (time2 - time1) > 0:
        # Calcula los FPS
        frames_per_second = 1.0 / (time2 - time1)
        # Escribe el número calculado los FPS.
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),3)
    # Actualiza el tiempo de fotograma anterior a este tiempo de marco.
    # Como este marco se convertirá en el marco anterior en la próxima iteración.
    time1 = time2
    # Muestra el frame.
    cv2.imshow('Subway Surfers Detector de posicion', frame)
    # Espera 1 ms. Si se presiona una tecla, recupere el código ASCII de la tecla.
    k = cv2.waitKey(1) & 0xFF
    # Comprueba si se presiona 'ESC' y termina el bucle.
    if (k == 27):
        break
# Suelta el objeto VideoCapture y cierra las ventanas.
video_camara.release()
cv2.destroyAllWindows()
