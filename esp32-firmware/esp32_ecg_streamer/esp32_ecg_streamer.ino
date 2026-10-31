#include <WiFi.h>
#include <HTTPClient.h>

// ================= আপনার আসল ওয়াই-ফাই রাউটার =================
const char* WIFI_SSID     = "Lucky 13";
const char* WIFI_PASSWORD = "qwertyuiop";

// ক্লাউড ব্যাকএন্ড লাইভ টেলিমেট্রি এন্ডপয়েন্ট
const char* SERVER_URL    = "https://sahayak-lkhm.onrender.com/api/ecg/stream";
const char* DEVICE_ID     = "ESP32_AD8232_REAL";

// এনালগ পিন (ESP32-এর VP / GPIO 36)
const int ECG_PIN = 36; 

const int SAMPLE_RATE_HZ = 100; // 100 Hz মেডিকেল স্যাম্পলিং
const unsigned long sampleIntervalMicros = 1000000 / SAMPLE_RATE_HZ; // 10000 µs

// ৫০ স্যাম্পল ব্যাচ (প্রতি ০.৫ সেকেন্ডে ১ বার ক্লাউডে যাবে)
const int BATCH_SIZE = 50; 
float ecgBuffer[BATCH_SIZE];
int bufferIndex = 0;

unsigned long lastSampleMicros = 0;
float dcBaseline = 2048.0;
float prevRaw = 2048.0;

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12); // ESP32 12-bit ADC (0 - 4095)
  analogSetAttenuation(ADC_11db); // Full-scale voltage (0 - 3.3V)

  Serial.println("\n--- সহায়ক IoT: বড় ও স্পষ্ট P-Q-R-S-T ইসিজি স্ট্রিমিং ---");
  
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  Serial.print("Lucky 13 রাউটারে কানেক্ট হচ্ছে...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n🎉 ওয়াই-ফাই সফলভাবে কানেক্টেড!");
  Serial.print("ESP32 আইপি: ");
  Serial.println(WiFi.localIP());
}

void sendBatchToServer() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.setTimeout(1200);
    http.addHeader("Content-Type", "application/json");

    String json = "{\"device_id\":\"" + String(DEVICE_ID) + "\",\"samples\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
      json += String(ecgBuffer[i], 3);
      if (i < BATCH_SIZE - 1) json += ",";
    }
    json += "]}";

    int httpCode = http.POST(json);
    http.end();
  }
}

void loop() {
  unsigned long currentMicros = micros();
  if (currentMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros = currentMicros;

    int raw = analogRead(ECG_PIN);

    // ১. ৫০ Hz নয়েজ ফিল্টার
    float filtered = 0.5f * ((float)raw + prevRaw);
    prevRaw = (float)raw;

    // ২. ডিসি বায়াস রিমুভাল
    dcBaseline = (0.97f * dcBaseline) + (0.03f * filtered);
    float acSignal = filtered - dcBaseline;

    // ৩. ৩x হাই-গেইন স্কেলিং (P-Q-R-S-T বড় ও স্পষ্ট দেখানোর জন্য /35.0f)
    float realEcg = acSignal / 35.0f;
    realEcg = constrain(realEcg, -3.5f, 3.5f);

    ecgBuffer[bufferIndex++] = realEcg;

    if (bufferIndex >= BATCH_SIZE) {
      sendBatchToServer();
      bufferIndex = 0;
    }
  }
}