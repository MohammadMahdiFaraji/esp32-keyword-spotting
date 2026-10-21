# My original course submission (January 2025)

These are the files I handed in for the final project of the Machine Learning course (Prof. Hamed Baghban). I left them as they were, so the problems I describe in the [main README](../../README.md) can be checked in the real code. The only change: I removed my student ID in two places.

The zip with the 28 test recordings is not included again. The 27 different clips are in [`data/own_recordings`](../../data/own_recordings); `yes_6.wav` was the same file as `yes_4.wav`.

Files:
- `MohammadMahdi_Faraji_final_ml_project.ipynb`: Colab notebook (data, MFCC, MLP training, evaluation, TFLite conversion)
- `fmcc_word_one.ipynb`: computes the MFCCs of one wav file, which I then pasted into the Arduino code
- `arduino ide-esp32/sketch_jan26a/sketch_jan26a.ino`: the ESP32 sketch
- `arduino ide-esp32/sketch_jan26a/modele.h`: the model as a C array
- `New Text Document.txt`: my notes for the submission (in Persian)

## What was wrong (and where)

| Problem | Where |
|---|---|
| random train/validation split, so the same speakers were in both | main notebook, `train_test_split(X_scaled, ...)` |
| StandardScaler fitted on all the data before the split (the cell is there twice) and never saved | main notebook, the two cells after the LabelEncoder |
| MFCCs padded to 50 frames, but 1 s clips only have 32, so frames 32-49 are always zero | main notebook, `max_pad_len=50` |
| learning rate 5e-5, the model was still improving after 100 epochs | main notebook, `learning_rate = 0.00005` |
| the features for the ESP32 are in a different order than in training (flatten first, then cut to 650) | `fmcc_word_one.ipynb`, second cell |
| no normalization on the ESP32 (raw MFCCs go into a model trained on scaled data) | `sketch_jan26a.ino`, `testPrediction()` |
| class names in the wrong order (the model outputs left, no, right, yes) | `sketch_jan26a.ino`, line 11 |
| missing `;`, the sketch does not compile like this | `sketch_jan26a.ino`, line 14 |
| `modele.h` is from a different training run than the results in the notebook | input layer is called `input_layer_1` in `modele.h` but `input_layer_3` in the notebook output |

Because of these bugs, the demo only looked correct by luck. For the "yes" clip in the sketch the model actually predicts "right", but the wrong class-name table prints it as "yes".
