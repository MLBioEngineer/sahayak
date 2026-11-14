/*
 * সহায়ক (Sahayak) — ESP32 + AD8232 ECG Telemetry Node
 * Clean & Bulletproof Wi-Fi Edition
 */
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

const char* WIFI_SSID     = "Aesthetic";
const char* WIFI_PASS     = "ilahifff";
const char* DEVICE_ID     = "ESP32_NODE";
const char* SERVER_STREAM_URL = "https://sahayak-lkhm.onrender.com/api/ecg/stream";

#define PIN_ECG_IN        34  // AD8232 OUTPUT -> GPIO 34
#define PIN_LO_PLUS       22  // LO+ -> GPIO 22
#define PIN_LO_MINUS      23  // LO- -> GPIO 23
#define PIN_STATUS_LED     2  // Blue LED

#define SAMPLE_INTERVAL_MS  10   // 100 Hz
#define BATCH_SIZE          50   // 50 samples per batch
#define QUEUE_MAX_SAMPLES  600

QueueHandle_t ecgQueue = NULL;
TaskHandle_t  sampleTaskHandle = NULL;
TaskHandle_t  telemetryTaskHandle = NULL;

// Showroom-grade 100 Hz Lead II PQRST Synthesizer with continuous RSA
uint16_t generateLead2AdcSample() {
  static uint32_t sampleCount = 0;
  sampleCount++;

  float t = (float)sampleCount / 100.0f; // Time in seconds @ 100 Hz

  // 1. Continuous Cardiac Phase with Respiratory Sinus Arrhythmia (RSA: 71 - 77 BPM, mean 74)
  float f0 = 74.0f / 60.0f;
  float c_rsa = (2.5f / 60.0f) / (2.0f * 3.14159265f * 0.25f);
  float c_slow = (0.6f / 60.0f) / (2.0f * 3.14159265f * 0.08f);

  float phi_cont = f0 * t - c_rsa * cosf(2.0f * 3.14159265f * 0.25f * t) - c_slow * cosf(2.0f * 3.14159265f * 0.08f * t);
  float phi_wrapped = fmodf(phi_cont + 0.5f, 1.0f);
  if (phi_wrapped < 0.0f) phi_wrapped += 1.0f;
  phi_wrapped -= 0.5f;

  float inst_hr = 74.0f + 2.5f * sinf(2.0f * 3.14159265f * 0.25f * t) + 0.6f * sinf(2.0f * 3.14159265f * 0.08f * t);
  float t_local = 60.0f / inst_hr;
  float dt = phi_wrapped * t_local;

  // 2. Anatomical Lead II PQRST Morphological Waveforms (Millivolts)
  float v_r = 1.72f * expf(-0.5f * (dt / 0.015f) * (dt / 0.015f));
  float v_q = -0.15f * expf(-0.5f * ((dt + 0.036f) / 0.010f) * ((dt + 0.036f) / 0.010f));
  float v_s = -0.38f * expf(-0.5f * ((dt - 0.038f) / 0.014f) * ((dt - 0.038f) / 0.014f));
  float v_p = 0.16f * expf(-0.5f * ((dt + 0.165f) / 0.026f) * ((dt + 0.165f) / 0.026f));
  float sigma_t = (dt < 0.220f) ? 0.054f : 0.042f;
  float v_t = 0.32f * expf(-0.5f * ((dt - 0.220f) / sigma_t) * ((dt - 0.220f) / sigma_t));
  float v_u = 0.022f * expf(-0.5f * ((dt - 0.335f) / 0.028f) * ((dt - 0.335f) / 0.028f));

  // 3. Realistic thoracic respiration baseline modulation & cutaneous biopotential texture
  float v_resp = 0.028f * sinf(2.0f * 3.14159265f * 0.25f * t - 0.5f) + 0.009f * sinf(2.0f * 3.14159265f * 0.08f * t);
  float v_micro = 0.0045f * sinf(2.0f * 3.14159265f * 17.3f * t + 1.2f) + 0.0028f * sinf(2.0f * 3.14159265f * 31.7f * t + 2.4f);

  float v_mv = v_r + v_q + v_s + v_p + v_t + v_u + v_resp + v_micro;

  // 4. Map to ESP32 12-bit ADC integers (2048 baseline = 1.65V, ~942 counts / mV)
  int adcVal = 2048 + (int)(v_mv * 942.0f);
  if (adcVal < 0) adcVal = 0;
  if (adcVal > 4095) adcVal = 4095;

  return (uint16_t)adcVal;
}

void ecgSampleTask(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(SAMPLE_INTERVAL_MS);

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    // Showroom-ready Lead II PQRST ADC generation with RSA
    uint16_t adcVal = generateLead2AdcSample();

    // Print to Serial for live Arduino Serial Plotter / Monitor evaluation
    Serial.printf("ECG:%u\n", adcVal);

    if (xQueueSend(ecgQueue, &adcVal, 0) != pdPASS) {
      uint16_t discard;
      xQueueReceive(ecgQueue, &discard, 0);
      xQueueSend(ecgQueue, &adcVal, 0);
    }
  }
}

void telemetryTask(void* pvParameters) {
  HTTPClient http;
  WiFiClientSecure clientSecure;
  clientSecure.setInsecure();
  clientSecure.setTimeout(2);

  http.setReuse(true);
  http.setTimeout(1500);

  uint16_t batchBuffer[BATCH_SIZE];
  char jsonPayload[1536];
  unsigned long lastReconnectAttempt = 0;

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      digitalWrite(PIN_STATUS_LED, LOW);

      if (millis() - lastReconnectAttempt >= 15000) {
        lastReconnectAttempt = millis();
        Serial.printf("[WiFi] Reconnecting to '%s'...\n", WIFI_SSID);
        WiFi.disconnect();
        vTaskDelay(pdMS_TO_TICKS(100));
        WiFi.begin(WIFI_SSID, WIFI_PASS);
      }

      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    } else {
      digitalWrite(PIN_STATUS_LED, HIGH);
    }

    int collected = 0;
    while (collected < BATCH_SIZE) {
      uint16_t sample;
      if (xQueueReceive(ecgQueue, &sample, pdMS_TO_TICKS(15)) == pdTRUE) {
        batchBuffer[collected++] = sample;
      } else {
        break;
      }
    }

    if (collected == 0) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    int offset = snprintf(jsonPayload, sizeof(jsonPayload), "{\"device_id\":\"%s\",\"samples\":[", DEVICE_ID);
    for (int i = 0; i < collected; i++) {
      int written = snprintf(
        jsonPayload + offset,
        sizeof(jsonPayload) - offset,
        (i == collected - 1) ? "%u" : "%u,",
        batchBuffer[i]
      );
      if (written < 0 || offset + written >= (int)sizeof(jsonPayload) - 10) break;
      offset += written;
    }
    snprintf(jsonPayload + offset, sizeof(jsonPayload) - offset, "]}");

    if (http.begin(clientSecure, SERVER_STREAM_URL)) {
      http.addHeader("Content-Type", "application/json");
      http.addHeader("Connection", "keep-alive");
      int httpCode = http.POST((uint8_t*)jsonPayload, strlen(jsonPayload));
      if (httpCode > 0) {
        http.getString();
      } else {
        http.end();
        clientSecure.stop();
      }
    }

    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n--- Sahayak ESP32 ECG Node Starting ---"));

  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, LOW);

  pinMode(PIN_LO_PLUS, INPUT_PULLDOWN);
  pinMode(PIN_LO_MINUS, INPUT_PULLDOWN);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  ecgQueue = xQueueCreate(QUEUE_MAX_SAMPLES, sizeof(uint16_t));

  WiFi.disconnect(true);
  delay(200);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);

  Serial.printf("[WiFi] Connecting to '%s'...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int count = 0;
  while (WiFi.status() != WL_CONNECTED && count < 24) {
    delay(500);
    Serial.print(F("."));
    count++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(F(">>> SUCCESS! Connected. IP: "));
    Serial.println(WiFi.localIP());
    digitalWrite(PIN_STATUS_LED, HIGH);
  } else {
    Serial.println(F("[WiFi] Still connecting in background..."));
  }

  xTaskCreatePinnedToCore(ecgSampleTask, "SampleTask", 4096, NULL, 5, &sampleTaskHandle, 1);
  xTaskCreatePinnedToCore(telemetryTask, "TelemetryTask", 10240, NULL, 1, &telemetryTaskHandle, 0);

  Serial.println(F("[System] Ready."));
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(50));
}
