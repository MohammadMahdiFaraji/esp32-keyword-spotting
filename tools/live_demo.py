# Live demo: say a word into the laptop microphone and the ESP32 tells you which word it was.
#
# The laptop records about 2 seconds, cuts out the 1 s around the word and computes the MFCCs
# (tools/kws_features.py, same as in the notebook). The MFCCs are sent over USB to the ESP32
# (firmware/kws_esp32), which runs the int8 model, turns on the LED of the word and sends the result back.
#
# usage (ESP32 connected, Arduino Serial Monitor closed):
#   pip install numpy pyserial sounddevice
#   python tools/live_demo.py                  press Enter and say a word
#   python tools/live_demo.py --quiz 24        the script tells you what to say and counts the correct answers
#   python tools/live_demo.py --wav data/own_recordings/*.wav
# options: --port COM5, --seconds 2.0, --save some_folder (keeps the recordings as wav files)

import argparse
import glob
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kws_features import CLASSES, KEYWORDS, SR, fit_to_1s, load_wav, mfcc_features

PROMPTS = {'silence': 'stay SILENT', 'unknown': 'say ANOTHER word (for example hello, stop, seven)'}


class Esp32:
    def __init__(self, port=None):
        import serial
        from serial.tools import list_ports

        if port is None:   # take the first USB serial port (CP210x / CH340 on most ESP32 boards)
            for p in list_ports.comports():
                if 'CP210' in p.description or 'CH340' in p.description or 'USB' in p.description:
                    port = p.device
                    break
        if port is None:
            sys.exit('no ESP32 found, connect it or use --port COMx')

        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = 115200
        self.ser.timeout = 0.05
        self.ser.dtr = False   # so the board is not reset when the port is opened
        self.ser.rts = False
        self.ser.open()
        self.buffer = b''

        # the board may still be busy with its test run after a reset, so try for a while
        print(f'ESP32 on {port}: ', end='', flush=True)
        end = time.time() + 20
        while time.time() < end:
            self.ser.write(b'P\n')
            if self.wait_for_line('READY', 1.5):
                print('ready')
                return
        sys.exit('no answer, is the sketch uploaded and the Serial Monitor closed?')

    def read_lines(self, timeout):
        # yields only complete lines (a line can arrive in two pieces)
        end = time.time() + timeout
        while time.time() < end:
            self.buffer += self.ser.read(self.ser.in_waiting or 1)
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                line = line.decode(errors='replace').strip()
                if line:
                    yield line

    def wait_for_line(self, start, timeout):
        for line in self.read_lines(timeout):
            if line.startswith(start):
                return line
        return None

    def classify(self, features):
        # returns (class, confidence, probabilities, time on the chip in us, gated)
        message = 'F ' + ','.join(f'{v:.9g}' for v in features.ravel()) + '\n'
        for attempt in range(2):   # try again once if the answer got lost
            self.ser.reset_input_buffer()
            self.buffer = b''
            self.ser.write(message.encode())
            line = self.wait_for_line(('RESULT', 'ERROR'), 5.0)
            if line is None:
                continue
            if line.startswith('ERROR'):
                raise RuntimeError(line)
            # RESULT <class> <confidence> <p0> ... <p5> <us> <gated>
            parts = line.split()
            probs = [float(v) for v in parts[3:-2]]
            return parts[1], float(parts[2]), probs, int(parts[-2]), parts[-1] == '1'
        raise TimeoutError('no answer from the ESP32')


def record(seconds):
    import sounddevice as sd
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]


def save_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    # same scale as load_wav (/ 32768), so the saved file is exactly the audio that was classified
    samples = np.clip(np.round(audio * 32768), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


def describe(name, conf, probs, us, gated):
    if gated:
        return f'ESP32: {name.upper():7s}         (too quiet, the network was not run)'
    order = np.argsort(probs)[::-1][:3]
    top3 = '  '.join(f'{CLASSES[k]} {probs[k]:.2f}' for k in order)
    return f'ESP32: {name.upper():7s} ({conf:.0%})   [{top3}]   {us / 1000:.0f} ms on the chip'


def listen_and_classify(esp, seconds, save_path=None):
    input('  press Enter, then speak ... ')
    print('  * recording *', flush=True)
    audio = record(seconds)
    if save_path:
        save_wav(save_path, audio)
    return esp.classify(mfcc_features(fit_to_1s(audio)))


def free_mode(esp, args):
    print('\nSay yes, no, left or right (or nothing, or another word). q + Enter to quit.')
    while True:
        if input('\nnext? [Enter = go, q = quit] ').strip().lower() == 'q':
            break
        save_path = None
        if args.save:
            save_path = Path(args.save) / f'live_{time.strftime("%Y%m%d_%H%M%S")}.wav'
        print('  ' + describe(*listen_and_classify(esp, args.seconds, save_path)))


def quiz(esp, args):
    # every class the same number of times, in random order
    targets = (CLASSES * (args.quiz // len(CLASSES) + 1))[:args.quiz]
    random.shuffle(targets)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    print(f'\nQuiz with {args.quiz} rounds. Do what is shown after you press Enter.')
    for i, target in enumerate(targets, 1):
        print(f'\n[{i}/{args.quiz}]  ' + PROMPTS.get(target, 'say:  ' + target.upper()))
        save_path = Path(args.save) / f'quiz_{target}_{i:02d}.wav' if args.save else None
        result = listen_and_classify(esp, args.seconds, save_path)
        # the target is only used here for counting, the ESP32 never gets it
        confusion[CLASSES.index(target), CLASSES.index(result[0])] += 1
        print('  ' + describe(*result) + ('   OK' if result[0] == target else '   WRONG'))
    print_confusion(confusion)


def wav_mode(esp, args):
    files = []
    for pattern in args.wav:
        files += glob.glob(pattern)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for f in sorted(files):
        result = esp.classify(mfcc_features(fit_to_1s(load_wav(f))))
        label = Path(f).stem.split('_')[0]   # file names are like left_3.wav
        verdict = ''
        if label in CLASSES:
            verdict = 'OK' if result[0] == label else 'WRONG'
            confusion[CLASSES.index(label), CLASSES.index(result[0])] += 1
        print(f'{Path(f).name:16s} {describe(*result)}   {verdict}')
    if confusion.sum() > 0:
        print_confusion(confusion)


def print_confusion(confusion):
    correct = np.trace(confusion)
    total = confusion.sum()
    print(f'\nESP32 accuracy: {correct}/{total} ({correct / total:.1%})')
    print('rows = what was asked, columns = what the ESP32 said')
    print('         ' + ''.join(f'{c:>8s}' for c in CLASSES))
    for c, row in zip(CLASSES, confusion):
        if row.sum() > 0:
            print(f'{c:>8s} ' + ''.join(f'{v:8d}' for v in row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live keyword spotting demo with the ESP32')
    parser.add_argument('--port', help='serial port, for example COM5 (default: first USB serial port)')
    parser.add_argument('--seconds', type=float, default=2.0, help='recording length in seconds')
    parser.add_argument('--quiz', type=int, help='number of quiz rounds')
    parser.add_argument('--wav', nargs='+', help='classify wav files (16 kHz mono) instead of the microphone')
    parser.add_argument('--save', help='folder where the recordings are saved')
    args = parser.parse_args()

    esp = Esp32(args.port)
    try:
        if args.wav:
            wav_mode(esp, args)
        elif args.quiz:
            quiz(esp, args)
        else:
            free_mode(esp, args)
    except KeyboardInterrupt:
        print()
