#include <WiFi.h>
#include <HTTPClient.h>

// ================= আপনার ওয়াই-ফাই রাউটার =================
const char* WIFI_SSID     = "Lucky 13";
const char* WIFI_PASSWORD = "qwertyuiop";

// ক্লাউড ব্যাকএন্ড লাইভ টেলিমেট্রি এন্ডপয়েন্ট
const char* SERVER_URL    = "https://sahayak-lkhm.onrender.com/api/ecg/stream";
const char* DEVICE_ID     = "ESP32_AD8232_REAL";

// এনালগ পিন (ESP32-এর VP / GPIO 36)
const int ECG_PIN = 36; 

const int SAMPLE_RATE_HZ = 100; // 100 Hz মেডিকেল স্যাম্পলিং
const unsigned long sampleIntervalMicros = 10000; // 10000 µs (10 ms)

// ১০০ স্যাম্পল = ঠিক ১.০ সেকেন্ডের নিখুঁত উইন্ডো (মেমোরি জ্যাম হবে না)
const int BATCH_SIZE = 100; 
float ecgBuffer[BATCH_SIZE];
int bufferIndex = 0;

unsigned long lastSampleMicros = 0;
float dcBaseline = 2048.0;
float prevRaw = 2048.0;

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.println("\n--- সহায়ক IoT: একটানা নন-স্টপ ইসিজি স্ট্রিমিং ---");
  
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
    http.setTimeout(1500); // টাইমআউট যাতে কখনো হ্যাং না হয়
    http.addHeader("Content-Type", "application/json");

    // কমপ্যাক্ট সিঙ্গেল-ফ্রেম পেলোড (দ্রুত ডেলিভারি)
    String json = "{\"device_id\":\"" + String(DEVICE_ID) + "\",\"samples\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
      json += String(ecgBuffer[i], 2);
      if (i < BATCH_SIZE - 1) json += ",";
    }
    json += "]}";

    int httpCode = http.POST(json);
    if (httpCode > 0) {
      Serial.println("🟢 [১০০ স্যাম্পল ক্লাউডে সফল]");
    } else {
      Serial.print("⚠️ নেটওয়ার্ক রিট্রাই... কোড: ");
      Serial.println(httpCode);
    }
    http.end();
  }
}

void loop() {
  unsigned long currentMicros = micros();
  if (currentMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros = currentMicros;

    int raw = analogRead(ECG_PIN);

    // ১. ৫০ Hz মেইনস নয়েজ ফিল্টার
    float filtered = 0.5f * ((float)raw + prevRaw);
    prevRaw = (float)raw;

    // ২. ডিসি বায়াস রিমুভাল
    dcBaseline = (0.97f * dcBaseline) + (0.03f * filtered);
    float acSignal = filtered - dcBaseline;

    // ৩. ৩x হাই-গেইন স্কেলিং (P-Q-R-S-T বড় ও উঁচু দেখানোর জন্য /35.0f)
    float realEcg = acSignal / 35.0f;
    realEcg = constrain(realEcg, -3.5f, 3.5f);

    ecgBuffer[bufferIndex++] = realEcg;

    // প্রতি ১০০ স্যাম্পল (১ সেকেন্ড) পর পর ক্লাউডে পাঠানো
    if (bufferIndex >= BATCH_SIZE) {
      sendBatchToServer();
      bufferIndex = 0;
    }
  }
}