# Indica la estructura de hiperparametros de los modelos pre-entrenados: https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml
# Para descargar la arquitectura y pesos usados aquí, en el siguiente link:
 ## https://drive.google.com/file/d/1F54gAXi7MwnaH3_qR4hAq5kBYoAJ3Ho7/view?usp=share_link
    
import cv2
import sys
import numpy as np

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    # blob: Poner la imagen en el estado ideal para que la red pueda realizar predicciones sobre ella.
        #* La sustracción de la media se utiliza para reducir el rango de valores de los píxeles y, en consecuencia, disminuir el impacto de los cambios de iluminación en las imágenes que componen nuestro dataset.
        #* https://datasmarts.net/es/como-funciona-cv2-dnn-blobfromimage-en-opencv/
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()
    # print(np.array(list(detections)).shape) : (1,1,200,7)
      #? shape[2]: Cada box detectada tiene 7 numeros
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            # Calcula el ancho y alto de una cadena de texto.
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Devuelve el tiempo total para la inferencia y los tiempos (en pasos) para las capas. 
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
