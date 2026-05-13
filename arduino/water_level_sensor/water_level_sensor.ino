#include "BluetoothSerial.h"
#include <TinyGPS++.h>

#include <WiFi.h>
#include <HTTPClient.h>
BluetoothSerial SerialBT;
TinyGPSPlus gps;

const char* ssid = "WIFI_ID";
const char* password = "WIFI_PW";
const int waterSensorPin = 34;

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
}

void loop() {

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

    HTTPClient http;

    // 서버 주소
    http.begin("SERVER_IP:5000/data");

    http.addHeader("Content-Type", "application/json");

    // JSON 데이터 생성
    String jsonData =
        "{\"water\": " + String(waterValue) +
        ", \"lat\": " + String(lat, 6) +
        ", \"lng\": " + String(lng, 6) + "}";
    // POST 요청
    int responseCode = http.POST(jsonData);

    Serial.print("응답 코드: ");
    Serial.println(responseCode);

    http.end();
  }

  delay(3000); //1분 : 6만
}