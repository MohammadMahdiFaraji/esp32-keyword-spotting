# How much to trust the numbers

A single accuracy number can hide a lot. Here is what each result really measures, and where it might be a bit too optimistic.

## The test set is the fair one
The test set has 2,467 clips from 233 speakers. None of them appear in the training or validation data, and I didn't use this set for any decision. Early stopping (deciding when to stop training) and the choice between the MLP and the CNN were both made on the validation set. So the test numbers are the most honest ones in this project.

## Our own recordings and the live test are a good sign, not a precise measurement
These clips were never used for training. Still, they influenced some of my choices:
- I added the "1 second around the word" step after I saw that three of my clips start late.
- I chose the safety margin of the loudness check after looking at how quiet my laptop's silence is.
- I picked the 6-class model after the live quiz.

So these numbers are probably a little too good. They are also based on very few clips. For example, 25 correct out of 27 sounds like 93 %, but with so few clips the real value could be anywhere between about 76 % and 98 % (95 % confidence interval).

One more detail: the live-test numbers come from playing the saved recordings to the board again, not from the live session itself.

## Silence
The silence clips in the test set are cut from the same six background-noise recordings as the training clips. They come from different parts of those recordings, so no piece is used twice. Even so, they are the same kinds of noise, so the 95 % for silence is probably optimistic for a new room. I haven't tested in a really noisy place.

## ESP32 vs. Python
For 27 of the 28 stored clips, the board gives exactly the same answer as the computer. The exception is `left_7`, which sits right on the border between two words. The computer says "no" with 50 % confidence, and the board says "left" with 63 %. The ESP32 uses its own optimized math routines for 8-bit numbers, and they round slightly differently.

To be sure the difference doesn't come from a bug in my code, I checked that the input numbers on the board are exactly the same as on the computer. They are, so the difference only comes from inside the network.

## Two training runs don't give exactly the same model
Training on a GPU isn't perfectly repeatable, so every run gives a slightly different model. In two runs with the same settings, the CNN reached 93.3 % and 93.7 % on the test set. Differences smaller than about half a percentage point don't mean much.
