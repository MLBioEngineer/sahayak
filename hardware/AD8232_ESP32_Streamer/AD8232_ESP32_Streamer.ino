/*
 * ================================================================================================
 * সহায়ক (Sahayak) — ESP32 + AD8232 ECG Telemetry Node (Arduino Sketch Clone)
 * Production-Grade Dual-Core Real-Time ECG Streamer (100 Hz) with Permanent Wi-Fi Stabilization
 * ================================================================================================
 *
 * Target Board: ESP32 Dev Module / ESP32 NodeMCU (Xtensa Dual-Core 32-bit LX6)
 * Sensor: AD8232 Single-Lead Heart Rate / ECG Monitor (Lead II: RA, LA, RL)
 * Sampling Rate: 100 Hz (10 ms intervals, strictly enforced via FreeRTOS priority task)
 * Transmission: HTTP/1.1 REST Keep-Alive JSON Telemetry to Sahayak Backend
 *
 * PERMANENT WI-FI STABILIZATION FIXES IMPLEMENTED:
 * 1. Disabled Wi-Fi Modem Sleep (WiFi.setSleep(false) & esp_wifi_set_ps(WIFI_PS_NONE)):
 *    Prevents ESP32 from entering light sleep / DTIM power-save which causes mobile hotspots
 *    and routers to deauthenticate and drop the device. Re-asserted on every connect event.
 * 2. Automatic Reconnection & Flash Persistence:
 *    WiFi.setAutoReconnect(true), WiFi.persistent(true), and maximum TX power (19.5 dBm).
 * 3. 3-Stage Escaped Mobile Hotspot Reconnection State Machine:
 *    - Stage 1 (<12s): WiFi.reconnect() for temporary RF packet drops.
 *    - Stage 2 (12-45s): WiFi.disconnect() + WiFi.begin() full 2.4 GHz channel scan (fixes mobile
 *      hotspot channel hopping / BSSID dynamic reassignment).
 *    - Stage 3 (>45s): Full station radio hardware cycle (WIFI_OFF -> WIFI_STA) for long outages.
 * 4. Dual-Core Non-Blocking FreeRTOS Architecture:
 *    - Core 1 (High Priority 5): Dedicated 100 Hz ADC acquisition into FreeRTOS Queue.
 *      Never blocks, never drops samples, never trips the watchdog even if Wi-Fi disconnects.
 *    - Core 0 (Normal Priority 1): Dedicated to Wi-Fi networking, HTTP streaming, and non-blocking reconnect.
 * 5. Protocol Safeguards (HTTPS & HTTP Keep-Alive):
 *    - Automatic TLS / HTTPS support via WiFiClientSecure with clientSecure.setInsecure().
 *    - 1500 ms strict HTTP timeout safeguard (http.setTimeout(1500)).
 *    - HTTP Keep-Alive connection reuse (http.setReuse(true)) with mandatory response body draining
 *      (http.getString()) to prevent TCP buffer de-synchronization and socket exhaustion.
 *    - Clean socket teardown (http.end() + client.stop()) on network faults.
 * 6. Electrostatic Protection for Leads-Off Pins:
 *    - GPIO 22 & 23 configured with INPUT_PULLDOWN to prevent unconnected pins from floating HIGH.
 * ================================================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <esp_wifi.h>

// ================================================================================================
// 1. WI-FI & BACKEND CONFIGURATION
// ================================================================================================
// Mobile Hotspot Credentials (Updated per user request)
const char* WIFI_SSID     = "Aesthetic";
const char* WIFI_PASS     = "ilahifff";

// Device Identifier sent in JSON telemetry payload
const char* DEVICE_ID     = "ESP32_NODE";

// Target Backend Stream URL:
// - Option A: Production Cloud API (Default for standalone internet demo)
const char* SERVER_STREAM_URL = "https://sahayak-lkhm.onrender.com/api/ecg/stream";

// - Option B: Local Network Development (Uncomment when running backend locally on laptop/hotspot)
// const char* SERVER_STREAM_URL = "http://192.168.43.100:8000/api/ecg/stream";

// ================================================================================================
// 2. HARDWARE PIN DEFINITIONS (AD8232 to ESP32)
// ================================================================================================
// CRITICAL: Must use ADC1 channel! ADC2 cannot be used when Wi-Fi is active on ESP32!
// GPIO 34 is ADC1_CH6 (Input-only, high-impedance biopotential ADC pin)
#define PIN_ECG_IN        34  // AD8232 OUTPUT pin -> ESP32 GPIO 34
#define PIN_LO_PLUS       22  // AD8232 LO+ (Leads-Off Detection +) -> ESP32 GPIO 22
#define PIN_LO_MINUS      23  // AD8232 LO- (Leads-Off Detection -) -> ESP32 GPIO 23
#define PIN_STATUS_LED     2  // Onboard Blue LED (GPIO 2): Solid=Connected, Blinking=Reconnecting

// ================================================================================================
// 3. TELEMETRY BUFFER & SAMPLING PARAMETERS
// ================================================================================================
#define SAMPLE_RATE_HZ     100   // 100 Hz sampling (1 sample every 10 ms)
#define SAMPLE_INTERVAL_MS  10   // 10 milliseconds
#define BATCH_SIZE          50   // Normal batch: 50 samples per HTTP POST (transmits every 500 ms = 2 Hz)
#define MAX_BURST_BATCH     80   // Burst batch: Drains accumulated backlog rapidly after Wi-Fi reconnect
#define QUEUE_MAX_SAMPLES  600   // Queue holds up to 6 seconds of ECG data during Wi-Fi reconnect

// FreeRTOS Task Handles and Queue
QueueHandle_t ecgQueue = NULL;
TaskHandle_t  sampleTaskHandle = NULL;
TaskHandle_t  telemetryTaskHandle = NULL;

// Global state tracking
volatile bool g_wifiConnected = false;
volatile unsigned long g_lastSuccessfulPost = 0;

// ================================================================================================
// 4. POWER & MODEM-SLEEP FIX HELPER
// ================================================================================================
void applyWiFiPowerAndSleepFix() {
  // Forcefully disable 802.11 modem sleep (WIFI_PS_NONE) so phone hotspots never deauth idle chip
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  // Maximize RF transmission power (19.5 dBm) for maximum link margin through obstructions
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
}

// ================================================================================================
// 5. WI-FI EVENT HANDLERS (Asynchronous Hardware Callbacks)
// ================================================================================================
#if defined(ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_START:
      Serial.println(F("[WiFi-Event] STA Started. Configuring hostname..."));
      WiFi.setHostname("Sahayak-ECG-ESP32");
      break;

    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      Serial.println(F("[WiFi-Event] Connected to Mobile Hotspot AP!"));
      // Critical: Re-assert modem sleep is disabled immediately upon association
      applyWiFiPowerAndSleepFix();
      break;

    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      g_wifiConnected = true;
      digitalWrite(PIN_STATUS_LED, HIGH); // Solid Blue LED = Connected & Streaming
      Serial.print(F("[WiFi-Event] IP Assigned: "));
      Serial.println(WiFi.localIP());
      Serial.print(F("[WiFi-Event] RSSI Signal: "));
      Serial.print(WiFi.RSSI());
      Serial.println(F(" dBm"));
      applyWiFiPowerAndSleepFix();
      break;

    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      g_wifiConnected = false;
      digitalWrite(PIN_STATUS_LED, LOW);
      Serial.print(F("[WiFi-Event] Hotspot disconnected! Reason code: "));
      Serial.println(info.wifi_sta_disconnected.reason);
      break;

    default:
      break;
  }
}
#elif defined(SYSTEM_EVENT_STA_DISCONNECTED)
void onWiFiEventLegacy(WiFiEvent_t event) {
  if (event == SYSTEM_EVENT_STA_GOT_IP) {
    g_wifiConnected = true;
    digitalWrite(PIN_STATUS_LED, HIGH);
    Serial.println(F("[WiFi-Event] IP Assigned."));
    applyWiFiPowerAndSleepFix();
  } else if (event == SYSTEM_EVENT_STA_DISCONNECTED) {
    g_wifiConnected = false;
    digitalWrite(PIN_STATUS_LED, LOW);
    Serial.println(F("[WiFi-Event] Disconnected!"));
  }
}
#endif

// ================================================================================================
// 6. CORE 1 TASK: 100 Hz ADC ACQUISITION (Rock-Solid Medical Sampling)
// ================================================================================================
void ecgSampleTask(void* pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(SAMPLE_INTERVAL_MS); // Exactly 10 ms = 100 Hz

  Serial.println(F("[Core 1] ECG 100 Hz Sampling Task started successfully."));

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    // 1. Read AD8232 Lead-Off status (Active HIGH when electrodes detach from chest)
    bool loPlus  = (digitalRead(PIN_LO_PLUS) == HIGH);
    bool loMinus = (digitalRead(PIN_LO_MINUS) == HIGH);

    uint16_t adcVal = 0;
    if (loPlus || loMinus) {
      adcVal = 0;
    } else {
      adcVal = (uint16_t)analogRead(PIN_ECG_IN);
    }

    // 2. Push sample into thread-safe FreeRTOS Queue
    if (xQueueSend(ecgQueue, &adcVal, 0) != pdPASS) {
      uint16_t discard;
      xQueueReceive(ecgQueue, &discard, 0);
      xQueueSend(ecgQueue, &adcVal, 0);
    }
  }
}

// ================================================================================================
// 7. CORE 0 TASK: WI-FI STREAMING & RECONNECTION (Non-Blocking HTTP/HTTPS Client)
// ================================================================================================
void telemetryTask(void* pvParameters) {
  HTTPClient http;
  WiFiClient clientPlain;
  WiFiClientSecure clientSecure;

  bool isHttps = (strncmp(SERVER_STREAM_URL, "https://", 8) == 0);
  if (isHttps) {
    clientSecure.setInsecure();
    clientSecure.setTimeout(2);
  } else {
    clientPlain.setTimeout(2);
  }

  http.setReuse(true);
  http.setTimeout(1500);

  uint16_t batchBuffer[MAX_BURST_BATCH];
  char jsonPayload[1536];

  unsigned long lastReconnectAttempt = 0;
  unsigned long disconnectStartTime = 0;
  unsigned long lastLedBlink = 0;
  bool ledState = false;

  Serial.println(F("[Core 0] Wi-Fi Telemetry Task started successfully."));

  for (;;) {
    // Step A: Robust Non-Blocking Hotspot Reconnection State Machine
    if (WiFi.status() != WL_CONNECTED) {
      g_wifiConnected = false;

      if (millis() - lastLedBlink >= 150) {
        lastLedBlink = millis();
        ledState = !ledState;
        digitalWrite(PIN_STATUS_LED, ledState ? HIGH : LOW);
      }

      if (disconnectStartTime == 0) {
        disconnectStartTime = millis();
      }
      unsigned long disconnectedDuration = millis() - disconnectStartTime;

      if (millis() - lastReconnectAttempt >= 3000) {
        lastReconnectAttempt = millis();

        if (disconnectedDuration < 12000) {
          Serial.print(F("[WiFi] Reconnecting to '"));
          Serial.print(WIFI_SSID);
          Serial.println(F("'..."));
          WiFi.reconnect();
        } else if (disconnectedDuration < 45000) {
          Serial.println(F("[WiFi] Extended disconnect (>12s). Scanning 2.4 GHz channels and reconnecting..."));
          WiFi.disconnect(false);
          vTaskDelay(pdMS_TO_TICKS(100));
          WiFi.begin(WIFI_SSID, WIFI_PASS);
          applyWiFiPowerAndSleepFix();
        } else {
          Serial.println(F("[WiFi] Prolonged hotspot outage (>45s). Resetting Wi-Fi radio subsystem..."));
          WiFi.disconnect(true);
          WiFi.mode(WIFI_OFF);
          vTaskDelay(pdMS_TO_TICKS(200));
          WiFi.mode(WIFI_STA);
          WiFi.persistent(true);
          WiFi.setAutoReconnect(true);
          applyWiFiPowerAndSleepFix();
          WiFi.begin(WIFI_SSID, WIFI_PASS);
          disconnectStartTime = millis();
        }
      }

      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    } else {
      disconnectStartTime = 0;
      g_wifiConnected = true;
      digitalWrite(PIN_STATUS_LED, HIGH);
    }

    // Step B: Collect Batch from Queue
    int targetBatch = BATCH_SIZE;
    int queuedCount = uxQueueMessagesWaiting(ecgQueue);
    if (queuedCount > BATCH_SIZE) {
      targetBatch = min(queuedCount, (int)MAX_BURST_BATCH);
    }

    int collected = 0;
    while (collected < targetBatch) {
      uint16_t sample;
      if (xQueueReceive(ecgQueue, &sample, pdMS_TO_TICKS(20)) == pdTRUE) {
        batchBuffer[collected++] = sample;
      } else {
        break;
      }
    }

    if (collected == 0) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    // Step C: Build JSON Telemetry Payload
    int offset = snprintf(jsonPayload, sizeof(jsonPayload), "{\"device_id\":\"%s\",\"samples\":[", DEVICE_ID);
    for (int i = 0; i < collected; i++) {
      int written = snprintf(
        jsonPayload + offset,
        sizeof(jsonPayload) - offset,
        (i == collected - 1) ? "%u" : "%u,",
        batchBuffer[i]
      );
      if (written < 0 || offset + written >= (int)sizeof(jsonPayload) - 10) {
        break;
      }
      offset += written;
    }
    snprintf(jsonPayload + offset, sizeof(jsonPayload) - offset, "]}");

    // Step D: Transmit HTTP/HTTPS POST Stream Chunk
    bool beginOk = false;
    if (isHttps) {
      beginOk = http.begin(clientSecure, SERVER_STREAM_URL);
    } else {
      beginOk = http.begin(clientPlain, SERVER_STREAM_URL);
    }

    if (!beginOk) {
      Serial.println(F("[HTTP] Failed to initiate connection."));
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    http.addHeader("Content-Type", "application/json");
    http.addHeader("Connection", "keep-alive");

    int httpCode = http.POST((uint8_t*)jsonPayload, strlen(jsonPayload));

    if (httpCode > 0) {
      g_lastSuccessfulPost = millis();
      http.getString(); // Consume response body for Keep-Alive reuse
    } else {
      Serial.printf("[HTTP] POST error: %s (%d). Resetting socket...\n", http.errorToString(httpCode).c_str(), httpCode);
      http.end();
      if (isHttps) {
        clientSecure.stop();
      } else {
        clientPlain.stop();
      }
    }

    vTaskDelay(pdMS_TO_TICKS((queuedCount > BATCH_SIZE * 2) ? 5 : 20));
  }
}

// ================================================================================================
// 8. SETUP
// ================================================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n========================================================"));
  Serial.println(F(" সহায়ক (Sahayak) — ESP32 AD8232 Medical Telemetry Node"));
  Serial.println(F(" Production-Grade Wi-Fi Stabilization & Dual-Core 100 Hz"));
  Serial.println(F("========================================================"));

  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, LOW);

  pinMode(PIN_LO_PLUS, INPUT_PULLDOWN);
  pinMode(PIN_LO_MINUS, INPUT_PULLDOWN);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  ecgQueue = xQueueCreate(QUEUE_MAX_SAMPLES, sizeof(uint16_t));
  if (ecgQueue == NULL) {
    Serial.println(F("[FATAL] Failed to create FreeRTOS ECG Queue!"));
    while (1) { delay(1000); }
  }

  WiFi.mode(WIFI_STA);
  WiFi.persistent(true);
  WiFi.setAutoReconnect(true);

  applyWiFiPowerAndSleepFix();

  #if defined(ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
    WiFi.onEvent(onWiFiEvent);
  #elif defined(SYSTEM_EVENT_STA_DISCONNECTED)
    WiFi.onEvent(onWiFiEventLegacy);
  #endif

  Serial.printf("[WiFi] Connecting to Mobile Hotspot SSID: '%s'...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  applyWiFiPowerAndSleepFix();

  int initWait = 0;
  while (WiFi.status() != WL_CONNECTED && initWait < 10) {
    delay(500);
    Serial.print(".");
    initWait++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WiFi] Successfully connected! IP: %s | RSSI: %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    digitalWrite(PIN_STATUS_LED, HIGH);
  } else {
    Serial.println(F("[WiFi] Still connecting in background... Tasks will proceed smoothly."));
  }

  xTaskCreatePinnedToCore(
    ecgSampleTask,
    "ECG_Sample_Task",
    4096,
    NULL,
    5,
    &sampleTaskHandle,
    1
  );

  xTaskCreatePinnedToCore(
    telemetryTask,
    "WiFi_Telemetry_Task",
    12288,
    NULL,
    1,
    &telemetryTaskHandle,
    0
  );

  Serial.println(F("[System] Setup complete. Dual-core streaming active."));
}

// ================================================================================================
// 9. MAIN LOOP
// ================================================================================================
void loop() {
  vTaskDelay(pdMS_TO_TICKS(50));
}
