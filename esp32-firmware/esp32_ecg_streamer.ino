/*
 * ==============================================================================
 * ESP32 + AD8232 Sleep Apnea ECG & EDR Streamer for সহায়ক (Sahayak)
 * Phase 6: Hardware Integration (Benchtop Evaluation & Live Streaming)
 * ==============================================================================
 * Pinout:
 *   AD8232 3.3V  -> ESP32 3V3
 *   AD8232 GND   -> ESP32 GND
 *   AD8232 OUT   -> ESP32 GPIO 36 (VP / ADC1_CH0)
 *   AD8232 LO+   -> ESP32 GPIO 22 (Optional Leads-off detection)
 *   AD8232 LO-   -> ESP32 GPIO 23 (Optional Leads-off detection)
 *   ESP32 DAC1   -> GPIO 25 (Optional PhysioNet Analog Loopback Simulation)
 * ==============================================================================
 * Physiological Principles:
 *   - Sampling Rate: 100 Hz (10 ms per sample) matching PhysioNet Apnea-ECG standard.
 *   - Telemetry Packet: Sends 100 samples every 1 second via HTTP POST to /api/ecg/stream.
 *   - Dual-Signal Recovery: The Sahayak AI system receives raw single-lead ECG and
 *     automatically extracts ECG-Derived Respiration (EDR) from R-peak amplitudes.
 *   - Apnea vs Normal Detection:
 *       * Normal Breathing: EDR shows regular sinusoidal breathing waves (~15 bpm).
 *       * Sleep Apnea / Breath-hold: EDR wave flatlines and vagal cardiac pause occurs.
 * ==============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi Configuration
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Sahayak Backend Live Stream Endpoint
const char* serverUrl = "https://sahayak-lkhm.onrender.com/api/ecg/stream";
const char* deviceId  = "ESP32_LEAD2_NODE";

// Hardware Configuration
const int ECG_PIN = 36;          // Analog Input from AD8232 OUTPUT
const int LO_PLUS = 22;          // Leads-off Positive
const int LO_MINUS = 23;         // Leads-off Negative
const int SAMPLE_RATE_HZ = 100;  // 100 Hz Sampling
const int BATCH_SIZE = 100;      // Transmit 100 samples per second

float ecgBuffer[BATCH_SIZE];
int bufferIndex = 0;
unsigned long lastSampleMicros = 0;
const unsigned long sampleIntervalMicros = 1000000 / SAMPLE_RATE_HZ; // 10,000 µs (10 ms)

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=======================================================");
  Serial.println("  Sahayak (সহায়ক) • ESP32 + AD8232 ECG Telemetry Node");
  Serial.println("=======================================================");

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  analogReadResolution(12); // 12-bit ADC (0 - 4095)

  // Connect to Wi-Fi
  Serial.printf("[Wi-Fi] Connecting to %s...", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 25) {
    delay(400);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[Wi-Fi] Connected successfully!");
    Serial.print("[Wi-Fi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[Wi-Fi] Connection timed out. Running in local acquisition mode.");
  }

  lastSampleMicros = micros();
}

void sendBatchToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // Construct JSON Payload
    String json = "{\"device_id\":\"" + String(deviceId) + "\",\"samples\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
      json += String(ecgBuffer[i], 3);
      if (i < BATCH_SIZE - 1) json += ",";
    }
    json += "]}";

    int httpResponseCode = http.POST(json);
    if (httpResponseCode > 0) {
      Serial.printf("[HTTP] Batch sent (Size: %d) -> Response: %d\n", BATCH_SIZE, httpResponseCode);
    } else {
      Serial.printf("[HTTP] Failed to send, error: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  }
}

void loop() {
  unsigned long currentMicros = micros();

  // Strict 100 Hz sampling timer
  if (currentMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros += sampleIntervalMicros;

    // Check leads-off
    bool leadsOff = (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1);
    float voltage = 0.0;

    if (!leadsOff) {
      int rawAdc = analogRead(ECG_PIN);
      // Map 12-bit ADC (0-4095) with ~1.65V virtual baseline to centered [-1.5, +2.0] mV scale
      voltage = ((float)rawAdc / 4095.0f * 3.3f) - 1.65f;
    }

    ecgBuffer[bufferIndex++] = voltage;

    // When 1 second buffer (100 samples) fills, transmit over Wi-Fi
    if (bufferIndex >= BATCH_SIZE) {
      sendBatchToServer();
      bufferIndex = 0;
    }
  }
}
