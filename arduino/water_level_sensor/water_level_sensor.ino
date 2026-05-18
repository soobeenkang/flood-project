#include "BluetoothSerial.h"
#include <TinyGPS++.h>

#include <PubSubClient.h>
#include <WiFi.h>
// #include <HTTPClient.h>
BluetoothSerial SerialBT;
TinyGPSPlus gps;

WiFiClient espClient;
PubSubClient client(espClient);
const char* ssid = "WIFI_ID";
const char* password = "WIFI_PW";

const char* mqtt_server = "SERVER_PORT";

const char* SENSOR_ID = "S001";

const int waterSensorPin = 34;

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
  // Serial2.begin(38400, SERIAL_8N1, 2, 23);
  Serial2.begin(9600,SERIAL_8N1,2,23);
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

  Serial.println("loop 시작");
    // 수위센서 값 읽기

  float lat = 0.0;
  float lng = 0.0;
  while (Serial2.available()) {

    gps.encode(Serial2.read());
  }

  int waterValue = analogRead(waterSensorPin);

  // 시리얼 출력
  Serial.print("수위: ");
  Serial.println(waterValue);
  Serial.print("위성 수: ");
  Serial.println(gps.satellites.value());

  Serial.print("GPS valid: ");
  Serial.println(gps.location.isValid());
  // GPS 위치 출력
  if (gps.location.isValid()) {

    lat = gps.location.lat();
    lng = gps.location.lng();

    Serial.print("위도: ");
    Serial.println(lat, 6);

    Serial.print("경도: ");
    Serial.println(lng, 6);

    SerialBT.print("수위: ");
    SerialBT.println(waterValue);

    SerialBT.print("위도: ");
    SerialBT.println(lat, 6);

    SerialBT.print("경도: ");
    SerialBT.println(lng, 6);
  }

  // WiFi 연결 상태 확인
  if (WiFi.status() == WL_CONNECTED) {
    String jsonData =
        "{\"sensor_id\": \"" + String(SENSOR_ID) +
        "\", \"water\": " + String(waterValue) +
        ", \"lat\": " + String(lat, 6) +
        ", \"lng\": " + String(lng, 6) + "}";

    client.publish("flood/data", jsonData.c_str());

    Serial.println("MQTT 전송 완료");
    Serial.println(jsonData);

  }

  delay(3000); //1분 : 6만
}