import cv2 as cv


def run():
    # Estructura de captura de video
    video = cv.VideoCapture(1)

    # leer desde la camara
    while True:
        ret, frame = video.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if i == 20:
            bg_frame_gray = frame_gray

        if i > 20:
            dif = cv.absdiff(frame_gray, bg_frame_gray)
            cv.imshow('dif', dif)

        # Mostrar frames del video
        cv.imshow('Fuente', frame)
        if cv.waitKey(100) & 0xFF == ord('s'):
            break

    # Eliminar ventana al salir
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()
