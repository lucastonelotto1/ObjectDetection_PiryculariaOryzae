# --- CONSULTA DIFCIL --- #

# Importaciones
import ultralytics
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Verificacin de GPU (opcional)
ultralytics.checks()

def analizar_imagen():
    # Cargar el modelo
    # Se asume que best.pt est en el mismo directorio o especificar ruta absoluta
    path_modelo = 'best.pt' 
    
    if not os.path.exists(path_modelo):
        print(f"Error: No se encontró el modelo en {os.path.abspath(path_modelo)}")
        return

    try:
        model = YOLO(path_modelo)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    print("Seleccione una imagen...")
    
    # Configurar ventana oculta de tkinter
    root = tk.Tk()
    root.withdraw() # Ocultar la ventana principal

    # Abrir selector de archivos
    filename = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imgenes", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    if not filename:
        print("No se seleccion ninguna imagen.")
        return

    print(f"\n Analizando: {filename}...")

    # Prediccin DIRECTA (YOLO se encarga de redimensionar y rellenar)
    # imgsz=640: Fuerza a YOLO a hacer el letterboxing a 640px
    try:
        results = model.predict(source=filename, imgsz=640, conf=0.25, save=False)
    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        return

    # Mostrar el resultado
    for r in results:
        # r.plot() devuelve la imagen con cajas en formato BGR (azul-verde-rojo)
        im_bgr = r.plot()

        # Convertir a RGB solo para que Matplotlib la muestre con colores reales
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title(f"Resultados para {os.path.basename(filename)}")
        plt.show()

        # Imprimir datos
        if len(r.boxes) == 0:
            print("  -> No se detect nada.")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            print(f"  -> {name}: {conf:.2f} ({conf*100:.1f}%)")

if __name__ == "__main__":
    analizar_imagen()