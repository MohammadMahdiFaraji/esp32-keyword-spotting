# What I changed compared to my course project

My original files are in [`legacy/original_course_submission`](../legacy/original_course_submission). That folder also has a list of where each problem is in the old code.

| Problem in my first version | What it caused | What I did |
|---|---|---|
| random 80/20 split | the same speakers were in train and validation, so the 93.6 % I reported was too optimistic | official split by speaker + a separate test set |
| MFCC padded to 50 frames, but 1 s clips only have 32 | 36 % of the inputs were always zero, so the model only used the first second | exactly 1 s of audio, padding on the audio |
| test clips cut from the start | in 3 of my clips the word came later | 1 s window around the word |
| one test file was there twice | 28 clips reported, only 27 different | duplicates are skipped (checked with a hash) |
| only the 4 words as classes | silence or any other word was always reported as a keyword | silence and unknown classes + energy gate |
| scaler fitted on all data and never saved | the ESP32 got raw MFCC values | mean / std from the training set, exported to the ESP32 |
| features for the ESP32 made in a different order | wrong input on the ESP32 | one feature function for everything |
| wrong order of the class names in the sketch | wrong labels | class names written to the header by the notebook |
| missing `;` in the sketch | it did not compile | fixed and checked with arduino-cli |
| float32 model, 361 KB | big and slow | int8, 58 KB |

## Why the old demo looked correct

For the demo I recorded a "yes", computed its MFCCs in Colab and pasted them into the sketch. Because of the bugs above, the old model actually predicted *right* for that clip. The class-name table in the sketch was in the wrong order, so it printed *yes*. Two bugs cancelled each other out, and the demo only looked correct by luck.
