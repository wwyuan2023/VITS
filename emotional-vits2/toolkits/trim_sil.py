
import os, sys
import fnmatch
import soundfile as sf
import librosa


def find_files(root_dir, query="*.wav"):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    return files

def trim_silence(infn, outfn):
    x, sr = librosa.load(infn, sr=8000)
    _, (xs, xe) = librosa.effects.trim(x, top_db=40)
    xs -= int(0.05 * sr)
    xe += int(0.05 * sr)
    if xs < 0: xs = 0
    x = x[xs:xe]
    x /= abs(x).max() * 2
    sf.write(outfn, x, sr, "PCM_16")


if __name__ == "__main__":
    in_wav_dir = sys.argv[1]
    out_wav_dir = sys.argv[2]
    i = 1
    for fn in find_files(in_wav_dir, "*.wav"):
        base = os.path.basename(fn)
        outfn = os.path.join(out_wav_dir, base)
        print(fn, outfn)
        trim_silence(fn, outfn)
        i += 1
    print(f"count={i}, Done!\n")
