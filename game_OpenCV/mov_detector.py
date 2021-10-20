import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1) as holistic:
    image = cv.imread("1.png")
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    result = holistic.process(image_rgb)


    # # rostro
    # mp_drawing.draw_landmarks(
    #     image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #     # color de los land marks (color, grosor linea, radio circulo)
    #     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
    #     # color de las conexiones
    #     mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))


    # Cuerpo (33 puntos)
    mp_drawing.draw_landmarks(
        image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        # color de los land marks (color, grosor linea, radio circulo)
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
        # color de las conexiones
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Mano izquieda (azul)
    mp_drawing.draw_landmarks(
        image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Mano derecha (verde)
    mp_drawing.draw_landmarks(
        image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))


    cv.imshow('image', image)
    cv.waitKey(0)

cv.destroyAllWindows()
