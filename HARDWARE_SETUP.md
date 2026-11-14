# সহায়ক (Sahayak) — ESP32 Wi-Fi Stabilization & Telemetry Setup Guide

> **মোবাইল হটস্পট আপডেট ও ওয়াইফাই স্থায়ী সমাধান:**
> - **SSID:** `Aesthetic`
> - **Password:** `ilahifff`
> - **Primary Firmware:** [`hardware/esp32_streamer/esp32_streamer.ino`](hardware/esp32_streamer/esp32_streamer.ino)
> - **PlatformIO Project:** [`esp32/`](esp32/)
> - **Live Cloud Stream:** `https://sahayak-lkhm.onrender.com/api/ecg/stream`
> - **Web Dashboard:** [https://sahayak-medai.vercel.app](https://sahayak-medai.vercel.app)

---

## ⚡ সংক্ষেপ: স্থায়ী ওয়াইফাই সমাধানের মূল বৈশিষ্ট্যসমূহ

| সমস্যা | কেন ঘটে? | আমাদের স্থায়ী সমাধান ও বাস্তবায়ন |
| :--- | :--- | :--- |
| **মোবাইল হটস্পট কিছুক্ষণ পর বিচ্ছিন্ন করে দেয়** | ESP32 ডিফল্ট মডেম স্লিপে গেলে ফোন হটস্পট আইডল ভেবে ডি-অথ করে | `WiFi.setSleep(false)` ও `esp_wifi_set_ps(WIFI_PS_NONE)` কানেকশন ইভেন্টেও পুনঃপ্রয়োগ করা হয় |
| **মোবাইল হটস্পট চ্যানেল পরিবর্তন করলে রিকানেক্ট হয় না** | `WiFi.reconnect()` কেবল পুরোনো চ্যানেলে খোঁজে | ৩-স্তরের এস্কেলেশন: ১২ সেকেন্ড পর সম্পূর্ণ ২.৪ GHz চ্যানেল স্ক্যান (`WiFi.begin`) করে |
| **HTTPS ক্লাউড সার্ভারে ডেটা যায় না** | সাধারণ `WiFiClient` পোর্ট ৪৪৩-এ প্লেইনটেক্সট পাঠায় | `WiFiClientSecure` + `clientSecure.setInsecure()` ইন্টিগ্রেশন |
| **কানেকশন রিউজ (Keep-Alive) এরর দেয়** | পূর্ববর্তী রেসপন্স বডি ড্রেন না করলে সকেট ডি-সিঙ্ক হয় | `http.getString()` দ্বারা রেসপন্স ক্লিয়ার করা হয় |
| **ডিসকানেক্ট হলে এডিসি ফ্রিজ বা ওয়াচডগ রিসেট** | সিঙ্গেল থ্রেডে ওয়াইফাই ড্রপ হলে ব্লক হয় | FreeRTOS ডুয়াল কোর: Core 1-এ ১০০ Hz স্যাম্পলিং এবং Core 0-এ নন-ব্লকিং ওয়াইফাই |
| **ইলেকট্রোড না লাগালে রিডিং শূন্যে আটকে থাকা** | ফ্লোটিং পিন নয়েজে LO+ হাই হয়ে যায় | GPIO 22/23-এ `INPUT_PULLDOWN` প্রোটেকশন |

---

## 📌 সার্কিট ডায়াগ্রাম (Wiring Diagram)

```
  [ AD8232 ECG Sensor ]               [ ESP32 Dev Module ]
       3.3V   ───────────────────────────> 3V3   (কখনোই 5V-এ লাগাবেন না)
       GND    ───────────────────────────> GND
      OUTPUT  ───────────────────────────> GPIO 34 (ADC1_CH6 — ওয়াইফাই চলাকালীন সম্পূর্ণ নিরাপদ)
       LO+    ───────────────────────────> GPIO 22 (Leads-Off + / পুলডাউন সুরক্ষিত)
       LO-    ───────────────────────────> GPIO 23 (Leads-Off - / পুলডাউন সুরক্ষিত)
```

> **টিপ:** যদি LO+ এবং LO- তার কানেক্ট না-ও করেন, ইন্টারনাল পুলডাউনের কারণে ইসিজি সিগন্যাল স্বাভাবিকভাবে কাজ করবে।

---

## 🚀 ফ্ল্যাশ করার সহজ ধাপ (Arduino IDE)

1. ল্যাপটপে **Arduino IDE** (v2.x বা v1.8.x) ওপেন করুন।
2. ফাইল ওপেন করুন: `hardware/esp32_streamer/esp32_streamer.ino`।
3. **Tools > Board > esp32 > ESP32 Dev Module** নির্বাচন করুন।
4. বোর্ড সেটিংস মিলিয়ে নিন:
   - **Upload Speed:** `921600` (বা সমস্যা হলে `115200`)
   - **CPU Frequency:** `240MHz (WiFi/BT)`
   - **Flash Frequency:** `80MHz`
   - **Flash Mode:** `QIO`
   - **Port:** আপনার ESP32-এর COM পোর্ট
5. **Upload (➡️)** বাটনে চাপ দিন।
6. আপনার মোবাইল ফোনের হটস্পট চালু করুন:
   - **Name (SSID):** `Aesthetic`
   - **Password:** `ilahifff`
   - **AP Band:** অবশ্যই **2.4 GHz** নির্বাচন করুন।
7. সিরিয়াল মনিটর (115200 baud) খুললে দেখতে পাবেন:
   ```text
   [Core 1] ECG 100 Hz Sampling Task started successfully.
   [Core 0] Wi-Fi Telemetry Task started successfully.
   [WiFi] Connecting to Mobile Hotspot SSID: 'Aesthetic'...
   [WiFi-Event] Connected to Mobile Hotspot AP!
   [WiFi-Event] IP Assigned: 192.168.43.xxx
   [System] Setup complete. Dual-core streaming active.
   ```
8. ESP32-এর অনবোর্ড ব্লু এলইডি (GPIO 2) স্থিরভাবে জ্বলে উঠবে এবং লাইভ ডেটা ক্লাউডে স্ট্রিম হতে থাকবে। ব্রাউজারে `https://sahayak-medai.vercel.app` ওপেন করে লাইভ মনিটরিং দেখুন!
