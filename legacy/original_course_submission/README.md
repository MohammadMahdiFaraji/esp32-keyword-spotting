# My original course submission (January 2025)

This is exactly what I handed in for the final project of the Machine Learning course (Prof. Hamed Baghban). I kept it unchanged on purpose. The [main README](../../README.md) describes the bugs I found later, and with the original files next to it, anyone can check those claims. The only thing I changed is that I removed my student ID in two places.

The zip file with the 28 test recordings is not repeated here. The 27 different clips are in [`data/own_recordings`](../../data/own_recordings). The 28th file, `yes_6.wav`, turned out to be an exact copy of `yes_4.wav`.

## What is in here

- `MohammadMahdi_Faraji_final_ml_project.ipynb`: the Colab notebook. It downloads the data, computes the features, trains the MLP, evaluates it and converts it to TensorFlow Lite.
- `fmcc_word_one.ipynb`: computes the features of a single recording, which I then pasted into the Arduino code by hand.
- `arduino ide-esp32/sketch_jan26a/sketch_jan26a.ino`: the code that ran on the ESP32.
- `arduino ide-esp32/sketch_jan26a/modele.h`: the model as a C array.
- `New Text Document.txt`: my notes for the submission (in Persian).

## Where the problems are

| Problem | Where to find it |
|---|---|
| Random train/validation split, so the same speakers ended up in both | main notebook: `train_test_split(X_scaled, ...)` |
| The scaler was fitted on all the data before the split (that cell is even there twice), and never saved | main notebook, the two cells after the `LabelEncoder` |
| The features were padded to 50 time steps, although a 1-second clip only has 32, so steps 32 to 49 were always zero | main notebook: `max_pad_len=50` |
| A very small learning rate (5e-5); the model was still improving after 100 epochs | main notebook: `learning_rate = 0.00005` |
| The features for the board were made in a different order (flattened first, then cut to 650 numbers) | `fmcc_word_one.ipynb`, second cell |
| No scaling on the board, so raw features went into a model trained on scaled ones | `sketch_jan26a.ino`, function `testPrediction()` |
| The names of the words in the wrong order (the model's order is left, no, right, yes) | `sketch_jan26a.ino`, line 11 |
| A missing `;`, so the sketch doesn't compile like this | `sketch_jan26a.ino`, line 14 |
| `modele.h` came from a different training run than the results printed in the notebook | the input layer is called `input_layer_1` in `modele.h`, but `input_layer_3` in the notebook output |

Together, these bugs explain why my demo still seemed to work. For the "yes" clip in the sketch, the model actually decided "right", and the wrong list of names printed "yes".
