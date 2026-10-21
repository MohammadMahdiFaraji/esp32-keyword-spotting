/*
  Keyword spotting on ESP32: yes / no / left / right (+ silence and unknown)

  The model and the three .h files come from notebooks/keyword_spotting.ipynb.
  After running the notebook, copy model_data.h, kws_config.h and test_samples.h into this folder.

  What the sketch does:
  - at startup it classifies the clips in test_samples.h (our recordings + one clip of digital silence)
    and prints the ESP32 result next to the Python result, so I can check that both are the same
  - after that it waits for commands on the serial port (115200 baud):
      T             run the test clips again (an empty line does the same)
      P             answers READY (used by tools/live_demo.py to check the connection)
      F v0,...,v636 classify one MFCC matrix sent from the laptop, answers with
                    RESULT <class> <confidence> <p0> ... <p5> <time in us> <gate 0/1>

  The MFCCs are computed on the laptop (tools/live_demo.py). I tried an analog microphone on the
  ESP32 during the course, but the ADC was too noisy.

  Board: ESP32 Dev Module (arduino-esp32 3.x), library: tflm_esp32
*/
#include <Arduino.h>
#include <tflm_esp32.h>

#include "model_data.h"
#include "kws_config.h"
#include "test_samples.h"

// one LED per keyword (yes, no, left, right). For silence / unknown all LEDs stay off.
const int kLedPins[kNumKeywords] = {5, 18, 19, 21};

constexpr int kArenaSize = 48 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void halt(const char* message) {
  Serial.println(message);
  while (true) delay(1000);
}

int numElements(TfLiteTensor* t) {
  int n = 1;
  for (int i = 0; i < t->dims->size; i++) n *= t->dims->data[i];
  return n;
}

// raw MFCC (49 x 13) -> probabilities of the 6 classes
// gated = true means the clip was too quiet and the network was not run at all
bool classify(const float* raw, float* probs, bool* gated) {
  // energy gate: loudest frame of the first coefficient (c0)
  float loudest = raw[0];
  for (int f = 1; f < kNumFrames; f++) {
    loudest = max(loudest, raw[f * kNumMfcc]);
  }
  *gated = loudest < kSilenceC0Threshold;
  if (*gated) {
    for (int k = 0; k < kNumClasses; k++) probs[k] = (k == kSilenceClass) ? 1.0f : 0.0f;
    return true;
  }

  // normalize with the values from training and quantize to int8 (same as in the notebook)
  for (int i = 0; i < kFeatureSize; i++) {
    int c = i % kNumMfcc;   // features are stored as [frame][coefficient]
    float x = (raw[i] - kFeatureMean[c]) / kFeatureStd[c];
    if (input->type == kTfLiteInt8) {
      int q = (int)rintf(x / input->params.scale) + input->params.zero_point;
      input->data.int8[i] = (int8_t)constrain(q, -128, 127);
    } else {
      input->data.f[i] = x;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) return false;

  for (int k = 0; k < kNumClasses; k++) {
    if (output->type == kTfLiteInt8) {
      probs[k] = (output->data.int8[k] - output->params.zero_point) * output->params.scale;
    } else {
      probs[k] = output->data.f[k];
    }
  }
  return true;
}

int argmax(const float* probs) {
  int best = 0;
  for (int k = 1; k < kNumClasses; k++) {
    if (probs[k] > probs[best]) best = k;
  }
  return best;
}

void showLed(int cls) {
  for (int k = 0; k < kNumKeywords; k++) {
    digitalWrite(kLedPins[k], k == cls ? HIGH : LOW);
  }
}

void runTestSet() {
  int correct = 0;
  int sameAsPython = 0;
  int numInvoked = 0;
  unsigned long totalTime = 0;

  Serial.println("\nfile             true     ESP32           Python");
  for (int s = 0; s < kNumTestSamples; s++) {
    float probs[kNumClasses];
    bool gated;
    unsigned long t0 = micros();
    if (!classify(kTestFeatures[s], probs, &gated)) halt("Invoke() failed");
    if (!gated) {
      totalTime += micros() - t0;
      numInvoked++;
    }

    // the prediction only comes from the model output, the labels are just used for counting
    int best = argmax(probs);
    if (best == kTestLabels[s]) correct++;
    if (best == kExpectedPred[s]) sameAsPython++;

    Serial.printf("%-16s %-8s %-7s %.2f    %-7s %.2f  %s%s%s\n", kTestNames[s], kClassNames[kTestLabels[s]],
                  kClassNames[best], probs[best], kClassNames[kExpectedPred[s]], kExpectedConf[s],
                  gated ? " (energy gate)" : "", best == kTestLabels[s] ? "" : " wrong",
                  best == kExpectedPred[s] ? "" : " (differs from Python!)");
    showLed(best);
    delay(250);
  }
  showLed(-1);

  Serial.printf("\nAccuracy on the stored clips: %d/%d (%.1f%%)\n", correct, kNumTestSamples,
                100.0f * correct / kNumTestSamples);
  Serial.printf("Same prediction as Python:    %d/%d\n", sameAsPython, kNumTestSamples);
  if (numInvoked > 0) {
    Serial.printf("Mean time per clip:           %lu us (normalize + quantize + inference)\n", totalTime / numInvoked);
  }
  Serial.println("\nSend T to run the test again, or use tools/live_demo.py for live audio.");
}

// "F v0,v1,...,v636" from tools/live_demo.py
void classifyFeatureLine(const String& line) {
  static float features[kFeatureSize];
  const char* p = line.c_str() + 1;   // skip the 'F'
  int n = 0;
  while (n < kFeatureSize) {
    char* end;
    float v = strtof(p, &end);
    if (end == p) break;   // no more numbers
    features[n] = v;
    n++;
    p = end;
    while (*p == ',' || *p == ' ') p++;
  }
  if (n != kFeatureSize) {
    Serial.printf("ERROR expected %d values, got %d\n", kFeatureSize, n);
    return;
  }

  float probs[kNumClasses];
  bool gated;
  unsigned long t0 = micros();
  if (!classify(features, probs, &gated)) {
    Serial.println("ERROR Invoke() failed");
    return;
  }
  unsigned long us = micros() - t0;

  int best = argmax(probs);
  showLed(best);
  Serial.printf("RESULT %s %.3f", kClassNames[best], probs[best]);
  for (int k = 0; k < kNumClasses; k++) Serial.printf(" %.3f", probs[k]);
  Serial.printf(" %lu %d\n", us, gated ? 1 : 0);
}

void setup() {
  Serial.setRxBufferSize(4096);   // one feature line is about 9 KB
  Serial.begin(115200);
  delay(2000);

  for (int k = 0; k < kNumKeywords; k++) {
    pinMode(kLedPins[k], OUTPUT);
    digitalWrite(kLedPins[k], LOW);
  }

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt("Model schema version does not match the library");

  // the ops used by the models. Keras Flatten / Reshape also add SHAPE, STRIDED_SLICE and PACK
  // (because of the dynamic batch size), without them AllocateTensors() fails
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensor_arena, kArenaSize);
  interpreter = &staticInterpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) halt("AllocateTensors() failed (missing op or arena too small)");

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (numElements(input) != kFeatureSize || numElements(output) != kNumClasses) {
    halt("Model input/output size does not match kws_config.h");
  }

  Serial.println("\n=== Keyword spotting on ESP32 ===");
  Serial.printf("Model: %s, %u bytes, input %s, arena used %u of %d bytes\n", KWS_MODEL_NAME, g_model_len,
                input->type == kTfLiteInt8 ? "int8" : "float32", (unsigned)interpreter->arena_used_bytes(), kArenaSize);
  runTestSet();
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.startsWith("F")) {
    classifyFeatureLine(line);
  } else if (line == "P") {
    Serial.println("READY");
  } else if (line.length() == 0 || line == "T" || line == "t") {
    runTestSet();
  } else {
    Serial.println("Commands: T = test clips, P = ping, F <637 values> = classify");
  }
}
