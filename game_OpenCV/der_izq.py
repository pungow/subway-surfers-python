import cv2 as cv
import mediapipe as mp
import pyautogui


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# leer la camara
video_camera = cv.VideoCapture(0)

# definimos tamaño de camara
camera_width = 640
camera_height = 480
video_camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_width)
video_camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_height)


def desplazamiento():
    hombro_der = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * frame_width)
    # hombro_der_y = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
    # cv.circle(frame, (hombro_der, hombro_der_y), 20, (255,255,255), -1)
    # cv.putText(frame, f'x:{hombro_der},y:{hombro_der_y}', (20,20), 1, 1, (255, 255, 255),2,cv.LINE_AA)
    hombro_izq = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * frame_width)
    cadera_der = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * frame_width)
    cadera_izq = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * frame_width)
    # y1 = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * frame_width)
    if hombro_der >= centro and cadera_der >= centro:
        cv.putText(frame, 'DER', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
        # pyautogui.press('righ')
    elif hombro_izq <= centro and cadera_izq <= centro:
        cv.putText(frame, 'IZQ', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
        # pyautogui.press('left')
    else:
        cv.putText(frame, 'centro', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)


with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1) as holistic:

    # # alto, ancho
    # sector_izq = int(camera_width * .4)
    # sector_der = int(camera_width * .4) + int(camera_width * .2)
    centro = int(camera_width / 2)
    while True:
        ret, frame = video_camera.read()
        # cambiamos a modo espejos
        frame = cv.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

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

        # texto (img, texto, coordenadas, fount(0-7), size, color, grosor, tipo de linea
        # cv.putText(frame, '0', (600, 500), 4, 1, (0, 255, 255), 2, cv.LINE_AA)

        # cv.line(frame, (int(frame_width/3), 0), (int(frame_width/3), frame_height), (0, 255, 0), 1)
        cv.line(frame, (centro, 0), (centro, frame_height), (0, 255, 0), 1)
        # cv.line(frame, (sector_izq, 0), (sector_izq, frame_height), (0, 255, 0), 1)
        # cv.line(frame, (sector_der, 0), (sector_der, frame_height), (0, 255, 0), 1)

        # obtenemos coordenadas de los puntos del torso
        if result.pose_landmarks is not None:
            desplazamiento()

        # mostar los frames
        cv.imshow('Frame', frame)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break

print(f'h: {frame_height}, w: {frame_width}')
video_camera.release()
cv.destroyAllWindows()
