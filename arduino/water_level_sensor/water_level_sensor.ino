#include "BluetoothSerial.h"
// #include <TinyGPS++.h>

#include <PubSubClient.h>
#include <WiFi.h>
// #include <HTTPClient.h>
BluetoothSerial SerialBT;
// TinyGPSPlus gps;

WiFiClient espClient;
PubSubClient client(espClient);
const char* ssid = "WIFI_ID";
const char* password = "WIFI_PASSWD";

const char* mqtt_server = "LOCALHOST";

const int sensorPins[] = {34/*,32,33*/};
const char* sensorIds[] = {
  "S001"/*,
  "S002",
  "S003"*/
};

const int SENSOR_COUNT = 1;

// const char* SENSOR_ID = "S001";
// const int waterSensorPin = 34;

void reconnectMQTT() {

  while (!client.connected()) {

    Serial.print("MQTT 연결 시도...");

    if (client.connect("ESP32Client")) {

      Serial.println("성공");

    } else {

      Serial.print("실패: ");
      Serial.println(client.state());

      delay(2000);
    }
  }
}
void setup() {

  Serial.begin(115200);
  // Serial2.begin(9600,SERIAL_8N1,2,23);
  SerialBT.begin("ESP32_GPS");

  analogReadResolution(12);

  WiFi.begin(ssid, password);

  Serial.print("WiFi 연결 중");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("연결 성공!");
  Serial.print("IP 주소: ");
  Serial.println(WiFi.localIP());
  client.setServer(mqtt_server, 1883);
}

void loop() {
  if (!client.connected()) {
    reconnectMQTT();
  }

  client.loop();

  for (int i=0;i<SENSOR_COUNT;i++){
    int waterValue = analogRead(sensorPins[i]);
    // WiFi 연결 상태 확인
    if (WiFi.status() == WL_CONNECTED) {
      Serial.print("수위: ");
      Serial.println(waterValue);
      
      String jsonData =
      "{\"sensor_id\": \"" + String(sensorIds[i]) +
      "\", \"water\": " + String(waterValue) + "}";
      
      client.publish("flood/data", jsonData.c_str());

      Serial.println("MQTT 전송 완료");
      Serial.println(jsonData);
    }
  }


  delay(3000); //1분 : 6만
}