import json
import time
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# Cargar el modelo preentrenado YOLOv8 nano (se descargará solo la primera vez)
print("[INICIALIZANDO] Cargando modelo YOLOv8...")
model = YOLO("yolov8n.pt")

# Parámetros del Broker MQTT (deben coincidir con el ESP32)
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SUB = "ucv/vigilancia/evento"
TOPIC_PUB_ALERTAS = "ucv/vigilancia/alertas"

def procesar_fotograma():
    """Simula la captura de la ESP32-CAM y ejecuta la inferencia con YOLOv8"""
    ruta_imagen = "muestra_intruso.jpg"
    print(f"\n[PROCESAMIENTO] Analizando captura de imagen: {ruta_imagen}")
    
    # Inferencia con YOLOv8
    inicio = time.time()
    results = model(ruta_imagen, verbose=False)
    latencia_inferencia = (time.time() - inicio) * 1000

    deteccion_persona = False
    max_confianza = 0.0

    for r in results:
        for box in r.boxes:
            clase_id = int(box.cls[0])
            confianza = float(box.conf[0])
            # La clase 0 en el dataset COCO corresponde a 'persona'
            if clase_id == 0:
                deteccion_persona = True
                if confianza > max_confianza:
                    max_confianza = confianza

    return deteccion_persona, max_confianza, latencia_inferencia

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[SERVIDOR ACTIVO] Conectado exitosamente al broker MQTT.")
        client.subscribe(TOPIC_SUB)
        print(f"[ESCUCHANDO] Suscrito al tópico: {TOPIC_SUB}")
    else:
        print(f"[ERROR] Código de conexión MQTT: {rc}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"\n[EVENTO RECIBIDO DESDE EDGE] {data}")
        
        if data.get("evento") == "movimiento_detectado":
            # Ejecuta la visión artificial
            hay_persona, confianza, latencia = procesar_fotograma()
            
            if hay_persona:
                reporte = {
                    "alerta": "INTRUSION CONFIRMADA",
                    "objeto": "persona",
                    "confianza": round(confianza * 100, 2),
                    "latencia_ms": round(latencia, 2),
                    "origen": "Servidor IA YOLOv8"
                }
                print(f"[ALERTA GENERADA] Persona confirmada ({reporte['confianza']}%) en {reporte['latencia_ms']} ms")
                client.publish(TOPIC_PUB_ALERTAS, json.dumps(reporte))
                print(f"[RED MQTT] Alerta enviada al canal: {TOPIC_PUB_ALERTAS}")
            else:
                print("[FILTRO IA] Falsa alarma: Movimiento detectado pero sin persona en cuadro.")
    except Exception as e:
        print(f"[ERROR] Fallo al procesar el mensaje: {e}")

# Configuración del cliente MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"[CONECTANDO] Enlazando al broker {MQTT_BROKER}...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Mantener en ejecución constante
client.loop_forever()