# How much to trust the numbers

## Test set
Only the test set is a really independent result. It has 2,467 clips from 233 speakers that are not in train or validation, and it was not used for any decision. Early stopping and the choice between the MLP and the CNN were both made on the validation set.

## Our own recordings and the live test
These clips were never used for training, but they did influence some decisions:
- I added the window around the word after I saw that three of my clips start late.
- I chose the margin of the energy gate after looking at the volume of my laptop's silence.
- I picked the 6-class version after the live quiz.

So these numbers are probably a bit too good. With only 27 and 24 clips they are also not very precise. For example, 25 / 27 means somewhere between about 76 % and 98 % with 95 % confidence.

The live-test numbers come from playing the saved recordings to the ESP32 again, not from the live session itself.

## Silence
The silence test clips come from the same 6 noise files as the training clips, only from different parts of them. So the silence result is probably optimistic. I did not test in a really noisy room.

## ESP32 vs. Python
The ESP32 gives the same class as the TFLite interpreter in Python for 27 of the 28 stored clips. The exception is `left_7`, which is a borderline case: Python gives *no* with 0.50, the ESP32 gives *left* with 0.63. The ESP32 uses its own optimized int8 kernels, and they round a little differently.

To make sure the problem was not in my code, I checked that the int8 input on the ESP32 is exactly the same as in Python. So the difference only comes from inside the network.

## Training runs
Results change a little between training runs, because training on the GPU is not exactly deterministic. In two training runs with the same settings, the CNN had 93.3 % and 93.7 % test accuracy.
