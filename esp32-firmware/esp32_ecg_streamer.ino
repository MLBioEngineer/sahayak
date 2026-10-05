/*
 * ==============================================================================
 * ESP32 + AD8232 Sleep Apnea ECG Streamer for সহায়ক (Sahayak)
 * Phase 6: Hardware Integration (Benchtop & Live Streamer)
 * ==============================================================================
 * Pinout:
 *   AD8232 3.3V  -> ESP32 3V3
 *   AD8232 GND   -> ESP32 GND
 *   AD8232 OUT   -> ESP32 GPIO 36 (VP / ADC1_CH0)
 *   AD8232 LO+   -> ESP32 GPIO 22 (Optional Leads-off check)
 *   AD8232 LO-   -> ESP32 GPIO 23 (Optional Leads-off check)
 *   ESP32 DAC1   -> GPIO 25 (Simulated ECG Playback without patient)
 * ==============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi Credentials
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Sahayak Backend Endpoint on Render
const char* serverUrl = "https://sahayak-lkhm.onrender.com/api/ecg/stream";

const int ECG_PIN = 36;          // Analog input from AD8232
const int SAMPLE_RATE_HZ = 100;  // 100 Hz Sampling (Standard for PhysioNet Apnea-ECG)
const int BATCH_SIZE = 100;      // Send 1 batch per second

float ecgBuffer[BATCH_SIZE];
int bufferIndex = 0;
unsigned long lastSampleMicros = 0;
const unsigned long sampleIntervalMicros = 1000000 / SAMPLE_RATE_HZ; // 10,000 us (10 ms)

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- Starting Sahayak ECG Streamer ---");

  // Connect to Wi-Fi
  Serial.printf("Connecting to Wi-Fi: %s\n", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[OK] Connected to Wi-Fi! IP: " + WiFi.localIP().toString());
}

void sendBatchToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // Build JSON payload
    String json = "{\"device_id\":\"ESP32_NODE_01\",\"samples\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
      json += String(ecgBuffer[i], 3);
      if (i < BATCH_SIZE - 1) json += ",";
    }
    json += "]}";

    int httpCode = http.POST(json);
    if (httpCode > 0) {
      Serial.printf("[HTTP] Batch sent. Response: %d\n", httpCode);
    } else {
      Serial.printf("[HTTP] POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  }
}

void loop() {
  unsigned long currentMicros = micros();

  // Strict 100 Hz sampling interval (every 10 milliseconds)
  if (currentMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros = currentMicros;

    // Read analog value (ADC is 0-4095 for 0-3.3V)
    int rawValue = analogRead(ECG_PIN);
    // Convert to centered normalized mV range
    float voltage = ((float)rawValue / 4095.0f * 3.3f) - 1.65f;

    ecgBuffer[bufferIndex++] = voltage;

    // When 1 full second (100 samples) is collected, stream to server
    if (bufferIndex >= BATCH_SIZE) {
      sendBatchToServer();
      bufferIndex = 0;
    }
  }
}
