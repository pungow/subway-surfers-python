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


with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1) as holistic:

    # alto, ancho
    sector_izq = [0, 0, 256, 480]
    sector_centro = [256, 0, 384, 480]
    sector_der = [480, 0, 640, 480]
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
        cv.line(frame, (256, 0), (256, frame_height), (0, 255, 0), 1)
        cv.line(frame, (384, 0), (384, frame_height), (0, 255, 0), 1)
        # obtenemos coordenadas de los puntos del torso
        if result.pose_landmarks is not None:
            hombro_der = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * frame_width)
            hombro_izq = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * frame_width)
            cadera_der = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * frame_width)
            cadera_izq = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * frame_width)
        # y1 = int(result.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * frame_width)
            if hombro_der <= 256 and hombro_izq <= 256 and cadera_der <= 256 and cadera_izq <= 256:
                cv.putText(frame, 'IZQ', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
                pyautogui.press('righ')
            elif hombro_der >= 384 and hombro_izq >= 384 and cadera_der >= 384 and cadera_izq >= 384:
                cv.putText(frame, 'DER', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
                pyautogui.press('left')
            else:
                cv.putText(frame, 'centro', (50, 50), 4, 1, (0, 0, 255), 2, cv.LINE_AA)
            print(hombro_der, hombro_izq, cadera_der, cadera_izq)
        # mostar los frames
        cv.imshow('Frame', frame)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break

print(f'h: {frame_height}, w: {frame_width}')
video_camera.release()
cv.destroyAllWindows()
