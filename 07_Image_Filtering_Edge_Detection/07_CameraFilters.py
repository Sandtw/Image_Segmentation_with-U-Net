
import cv2
import sys
import numpy

PREVIEW  = 0   # Preview Mode
BLUR     = 1   # Blurring Filter
FEATURES = 2   # Corner Feature Detector
CANNY    = 3   # Canny Edge Detector

feature_params = dict( maxCorners = 500,  
                       qualityLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9)
                       
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = 'Camera Filters'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)
while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame,1)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        #? cv2.Canny(frame, minVal, maxVal): Algoritmo de detección de bordes
            # minVal: Valor de umbral inferior en Hysteresis Thresholding
        result = cv2.Canny(frame, 85, 150)
    elif image_filter == BLUR:
        #? cv2.blur(frame, ksize): Desenfocar una imagen utilizando el filtro de cuadro normalizado
            # ksize: Tamaño del kernel de desenfoque
        result = cv2.blur(frame, (13,13))
    elif image_filter == FEATURES:
         result = frame
         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         #? cv2.goodFeaturesToTrack(frame, **feature_params): Encuentra N esquinas más fuertes mediante Shi-Tomasi o Harris Corner Detector
            # img: La imagen debe ser una imagen en escala de grises
            # qualityLevel: Si la mejor esquina tiene una puntuacion de calidad = 1500, y qualityLevel = 0.01, se rechazan todas las esquinas detectadas con una puntuacion de calidad inferior a 15.
                #* Ordena las esquinas restantes según la calidad en orden descendente. Luego, la función toma la primera esquina más fuerte, descarta todas las esquinas cercanas en el rango de distancia mínima y devuelve N esquinas más fuertes.
            # minDistance: Distancia euclidiana mínima entre las esquinas detectadas.
         corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
         if corners is not None:
             for x, y in numpy.int0(corners).reshape(-1, 2):
                 cv2.circle(result, (x,y), 10, (0, 255 , 0), 1)

    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(win_name)
