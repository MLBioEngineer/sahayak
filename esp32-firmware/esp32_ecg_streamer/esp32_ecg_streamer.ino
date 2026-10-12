/*
 * ==============================================================================
 * ESP32 ECG Streamer (HIL Simulation Mode) for Sahayak
 * ==============================================================================
 * Connect a single jumper wire from GPIO 25 to GPIO 36 (VP).
 * ==============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi Credentials - REPLACE THESE
const char* ssid     = "Lucky 13";
const char* password = "qwertyuiop";

// Sahayak Backend Endpoint
const char* serverUrl = "https://sahayak-lkhm.onrender.com/api/ecg/stream";

const int ECG_PIN = 36;          // Reading fake signal
const int DAC_PIN = 25;          // Generating fake signal
const int SAMPLE_RATE_HZ = 100;
const int BATCH_SIZE = 100;

float ecgBuffer[BATCH_SIZE];
int bufferIndex = 0;
unsigned long lastSampleMicros = 0;
const unsigned long sampleIntervalMicros = 1000000 / SAMPLE_RATE_HZ;

int fakeEcgPhase = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- Starting Sahayak ECG Streamer (Simulation Mode) ---");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[OK] Connected! IP: " + WiFi.localIP().toString());
}

void sendBatchToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String json = "{\"device_id\":\"ESP32_SIM_01\",\"samples\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
      json += String(ecgBuffer[i], 3);
      if (i < BATCH_SIZE - 1) json += ",";
    }
    json += "]}";

    int httpCode = http.POST(json);
    if (httpCode > 0) {
      Serial.printf("[HTTP] Batch sent. Response: %d\n", httpCode);
    }
    http.end();
  }
}

void generateFakeECG() {
  // Generate a basic heartbeat-like wave for the DAC (0-255)
  fakeEcgPhase = (fakeEcgPhase + 1) % 100;
  int dacValue = 127; // Baseline
  if (fakeEcgPhase == 10) dacValue = 150; // P wave
  else if (fakeEcgPhase == 20) dacValue = 100; // Q wave
  else if (fakeEcgPhase == 22) dacValue = 255; // R wave (Peak)
  else if (fakeEcgPhase == 25) dacValue = 80;  // S wave
  else if (fakeEcgPhase == 40) dacValue = 160; // T wave
  dacWrite(DAC_PIN, dacValue);
}

void loop() {
  unsigned long currentMicros = micros();

  if (currentMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros = currentMicros;
    
    generateFakeECG(); // Create fake heartbeat on pin 25
    
    // Read the fake heartbeat on pin 36
    int rawValue = analogRead(ECG_PIN);
    float voltage = ((float)rawValue / 4095.0f * 3.3f) - 1.65f;
    ecgBuffer[bufferIndex++] = voltage;

    if (bufferIndex >= BATCH_SIZE) {
      sendBatchToServer();
      bufferIndex = 0;
    }
  }
}
