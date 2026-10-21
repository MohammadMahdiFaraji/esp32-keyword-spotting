# What I changed compared to my course project

When I came back to my course project, I first wanted to clean it up a bit. Instead I found a chain of bugs. Some of them made the numbers look better than they were. Others meant the board never got the input the model expected. Here is the full list, with what each problem caused and how I fixed it. My original files are in [`legacy/original_course_submission`](../legacy/original_course_submission), together with a list of where each problem sits in the old code.

| What was wrong | Why it mattered | What I did |
|---|---|---|
| The data was split into train and validation at random | The same people were in both parts. The model was partly tested on voices it already knew, so the 93.6 % I reported was too optimistic. | I use the official split of the dataset, where every speaker is in only one part, plus a separate test set |
| The features were padded to 50 time steps, but a 1-second clip only has 32 | More than a third of the input was always zero, so the model only ever looked at the first second | Exactly 1 second of audio; if a clip is shorter, the padding is added to the sound, not to the features |
| My own test clips were cut from the start | In 3 of my clips the word came after the first second, so the model heard silence | I cut out the 1 second around the word |
| One test file was in the folder twice | I reported 28 test clips, but only 27 were different | Exact copies are now found with a hash and skipped |
| The model only knew the four words | Silence, coughs or any other word were always reported as one of the four | Two extra classes, *silence* and *unknown*, plus a quick loudness check on the board |
| The scaling of the features was fitted on all the data and never saved | The board got raw, unscaled numbers, while the model was trained on scaled ones | The scaling values come from the training data only, and the notebook writes them into a header file for the board |
| The features for the board were made by a separate, slightly different piece of code | The numbers arrived in a different order than during training | One feature function is used for everything |
| The names of the words in the Arduino code were in the wrong order | The board printed the wrong word | The notebook writes the word list into a header file, so it always matches the model |
| A semicolon was missing in the sketch | It did not compile in its final form | Fixed, and every version is compiled before it goes on the board |
| The model was stored as 32-bit floats (361 KB) | Big and slow for a microcontroller | int8 model, 58 KB |

## Why the old demo still seemed to work

For the demo, I recorded a "yes", computed its features in Colab and pasted the numbers into the Arduino code. Because of the bugs above, the model on the board actually decided "right". But the table with the names of the words was in the wrong order, so the board printed "yes". Two mistakes cancelled each other out, and the demo looked correct purely by luck.
