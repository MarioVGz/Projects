import cv2

# Cargar el clasificador de cascada
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error: No se pudo cargar el clasificador de cascada")
    exit()

# Iniciar la captura de video desde la webcam (asegúrate de que el índice de la cámara es correcto)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

while True:
    # Leer una imagen del video
    ret, img = cap.read()
    if not ret:
        print("Error: No se pudo leer la imagen de la cámara")
        break
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Mostrar la imagen
    cv2.imshow('img', img)
    
    # Detectar si se está presionando la tecla ESC (ASCII 27)
    k = cv2.waitKey(30)
    if k == 27:  # 27 es el ASCII para ESC
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()