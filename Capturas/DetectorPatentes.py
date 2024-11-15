# Importar las librerias necesarias
import numpy as np
import torch
import cv2
import numpy
import pandas

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5','custom', path='M:/Programas en Python/Python/VisionPatentes/Capturas/model/patentes.pt', force_reload=True)


# Realizamos la Captura de la imagen

def capture_image():
    cap = cv2.VideoCapture(0) # Se usa '0' para cámaras integradas, o '1', '2' para cámaras externeas
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar la imagen.")
            return None

        # Se realiza la deteccion

        detect = model(frame)

        #Mostramos FPS
        cv2.imshow('Detector de patentes', np.squeeze(detect.render()))
        # Para salir del ciclo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()