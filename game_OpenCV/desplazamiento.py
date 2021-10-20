import cv2 as cv
import mediapipe as mp
import pyautogui


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# leer la camara
video_camera = cv.VideoCapture(0)

# definimos tamaño de camara
width = 640
height = 480
video_camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
video_camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)


def desplazamiento():
    # desplazamiento en eje x (horizontal)
    hombro_der = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * width)
    hombro_izq = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * width)

    if hombro_der > centro_x:
        cv.putText(frame, 'DER', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)

    elif hombro_izq < centro_x:
        cv.putText(frame, 'IZQ', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)

    else:
        cv.putText(frame, 'centro', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)

    # Desplazamiento en y (vertical)
    nariz = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * height)
    hombro_der_y = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * height)
    hombro_izq_y = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * height)
    # supere eje de referencia
    if nariz > ref_y:
        cv.putText(frame, 'abajo', (50, 100), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
    elif hombro_der_y < ref_y and hombro_izq_y < ref_y:
        cv.putText(frame, 'arriba', (50, 100), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame, 'no', (50, 100), 4, 1, (0, 0, 255), 2, cv.LINE_AA)


with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1) as holistic:

    centro_x = int(width / 2)
    ref_y = int(height * .3)

    while True:
        ret, frame = video_camera.read()
        # cambiamos a modo espejo
        frame = cv.flip(frame, 1)
        # deteccion delos puntos holistic
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = holistic.process(image_rgb)

        # Cuerpo (33 puntos)-
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            # color de los land marks (color, grosor linea, radio circulo)
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
            # color de las conexiones
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

        # dibujamos lineas de referencias
        cv.line(frame, (centro_x, 0), (centro_x, height), (0, 255, 0), 1)
        cv.line(frame, (0, ref_y), (width, ref_y), (255, 255, 255), 1)

        # obtenemos coordenadas de los puntos del torso
        if result.pose_landmarks is not None:
            desplazamiento()

        # mostar los frames
        cv.imshow('Frame', frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

video_camera.release()
cv.destroyAllWindows()
