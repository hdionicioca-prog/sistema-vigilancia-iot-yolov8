import cv2
import json
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# 1. Configuración de conexión MQTT
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_ALERTAS = "ucv/vigilancia/alertas"

cliente_mqtt = mqtt.Client()
print("[RED] Conectando al broker MQTT...")
cliente_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
cliente_mqtt.loop_start()

# 2. Cargar el modelo YOLOv8 ligero
print("[IA] Cargando modelo YOLOv8 (yolov8n.pt)...")
modelo = YOLO("yolov8n.pt")

# 3. Inicializar la cámara conectada
# El índice 0 es la cámara por defecto. Si tienes dos cámaras, prueba con 1.
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("[ERROR] No se pudo acceder a la cámara. Revisa permisos o desconecta/reconecta el USB.")
    exit()

tiempo_ultima_alerta = 0
INTERVALO_ENTRE_ALERTAS = 3.0  # Segundos de espera entre envíos

print("\n[ESTADO] Sistema activo. Pasa frente a la cámara. Presiona 'q' en la ventana para salir.")

while True:
    exito, cuadro = camara.read()
    if not exito:
        print("[ERROR] Fallo al capturar el cuadro de video.")
        break

    # Ejecutar detección con YOLOv8 en el cuadro actual
    resultados = modelo(cuadro, verbose=False)
    persona_detectada = False
    max_confianza = 0.0

    # Revisar las detecciones encontradas
    for res in resultados:
        for caja in res.boxes:
            clase = int(caja.cls[0])
            confianza = float(caja.conf[0])

            # En el dataset COCO de YOLOv8, la clase 0 corresponde a 'persona'
            if clase == 0 and confianza >= 0.50:
                persona_detectada = True
                if confianza > max_confianza:
                    max_confianza = confianza

                # Dibujar recuadro visual de intrusión
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                cv2.rectangle(cuadro, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cuadro, f"INTRUSO: {confianza * 100:.1f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Si se confirma la intrusión física y pasó el tiempo de espera mínimo
    tiempo_actual = time.time()
    if persona_detectada and (tiempo_actual - tiempo_ultima_alerta > INTERVALO_ENTRE_ALERTAS):
        payload = {
            "evento": "INTRUSION_CONFIRMADA",
            "objeto": "persona",
            "confianza": round(max_confianza * 100, 2),
            "hora": time.strftime("%H:%M:%S")
        }
        cliente_mqtt.publish(TOPIC_ALERTAS, json.dumps(payload))
        print(f"[ALERTA PUBLICADA EN MQTT] {payload}")
        tiempo_ultima_alerta = tiempo_actual

    # Mostrar la imagen en una ventana de escritorio
    cv2.imshow("Sistema de Vigilancia - Deteccion en Vivo", cuadro)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()
cliente_mqtt.loop_stop()
cliente_mqtt.disconnect()
print("[ESTADO] Sistema finalizado.")