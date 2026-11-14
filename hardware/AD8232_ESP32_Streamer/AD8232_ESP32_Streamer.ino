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

struct EcgSample {
  uint16_t adc;
  bool leadsOff;
};

QueueHandle_t ecgQueue = NULL;
TaskHandle_t  sampleTaskHandle = NULL;
TaskHandle_t  telemetryTaskHandle = NULL;

// Real biopotential sampling task: reads genuine microvolt Lead II ECG from AD8232 on GPIO 34
void ecgSampleTask(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(SAMPLE_INTERVAL_MS);

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    // Read real AD8232 Leads-Off detection pins (LO+ on GPIO 22, LO- on GPIO 23)
    bool loPlus = (digitalRead(PIN_LO_PLUS) == HIGH);
    bool loMinus = (digitalRead(PIN_LO_MINUS) == HIGH);

    // Read real biopotential analog voltage from AD8232 OUTPUT pin on GPIO 34
    uint16_t adcVal = (uint16_t)analogRead(PIN_ECG_IN);

    // Honest hardware-level Leads-Off check:
    // Triggered if LO+/LO- active OR ADC reading rails to 3.3V / GND rail saturation
    bool isRailed = (adcVal >= 3850 || adcVal <= 90);
    bool leadsOff = loPlus || loMinus || isRailed;

    EcgSample sample;
    sample.adc = adcVal;
    sample.leadsOff = leadsOff;

    // Print to Serial for live Arduino Serial Plotter / Monitor evaluation
    if (leadsOff) {
      Serial.printf("ECG:%u [LEADS_OFF]\n", adcVal);
    } else {
      Serial.printf("ECG:%u\n", adcVal);
    }

    if (xQueueSend(ecgQueue, &sample, 0) != pdPASS) {
      EcgSample discard;
      xQueueReceive(ecgQueue, &discard, 0);
      xQueueSend(ecgQueue, &sample, 0);
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
    }

    int collected = 0;
    int loCount = 0;
    while (collected < BATCH_SIZE) {
      EcgSample sample;
      if (xQueueReceive(ecgQueue, &sample, pdMS_TO_TICKS(15)) == pdTRUE) {
        batchBuffer[collected] = sample.adc;
        if (sample.leadsOff) loCount++;
        collected++;
      } else {
        break;
      }
    }

    if (collected == 0) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    bool batchLeadsOff = (loCount > collected / 4);

    // Visual feedback on status LED: solid ON when leads attached, gentle pulse when leads detached
    if (batchLeadsOff) {
      digitalWrite(PIN_STATUS_LED, (millis() % 500 < 100) ? HIGH : LOW);
    } else {
      digitalWrite(PIN_STATUS_LED, HIGH);
    }

    int offset = snprintf(
      jsonPayload,
      sizeof(jsonPayload),
      "{\"device_id\":\"%s\",\"leads_off\":%s,\"samples\":[",
      DEVICE_ID,
      batchLeadsOff ? "true" : "false"
    );
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

  ecgQueue = xQueueCreate(QUEUE_MAX_SAMPLES, sizeof(EcgSample));

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
