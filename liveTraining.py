import os
import time
import subprocess


"""Runs the tf.keras training script while handling 
and writing the output to an index.html file. Caddy 
is watching and proxying this file, so it gets live 
updates to the domain, showing live training stats.
"""

def purgeSlate(idx):
    """Delete the html page before a new training session."""
    # os.remove(idx)
    rep = os.path.join(os.path.dirname(idx), f"index_{int(time.time())}.txt")
    os.rename(idx, rep)

def liveTraining(script):
    """Generates lines of training process while training."""
    script = os.path.abspath(script)
    ln = 0
    try:
        training = subprocess.Popen(
            ["python3", "-u", f"{script}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in training.stdout:
            ln += 1
            yield line, ln

    except KeyboardInterrupt:
        return

def writeStats(idx, line, ln):
    """Writes lines of output given to the .html file as formatted html."""
    idx = "/Volumes/HomeXx/compuir/neat/cartigliaClub/index.html"
    header = '<div style="white-space: pre;">'
    tail = "</div>"

    if ln == 1:
        purgeSlate(idx)

    if line:
        if ln % 2 == 0:
            line = line + "\n"
        with open(idx, 'a') as f:
            f.write(header)
            f.write(line)
            f.write(tail)


def insertImage(idx, srcs):
    """Gets the training history plot to the index.html page."""
    dest = os.path.dirname(idx)
    srcd = os.path.dirname(srcs)

    for _dir in os.scandir(srcd):
        if os.path.isdir(_dir):
            if "handModel" in _dir.path:
                models.append(_dir)
    models.sort()
    select = models[-1]

    src = os.path.join(select, "training.png")

    with open(idx, 'a') as i:
        i.write(f"<img src='{src}' alt='Training/Validation Loss across Epochs'>")

def main():
    idx = "/Volumes/HomeXx/compuir/neat/cartigliaClub/index.html"
    script = "/Volumes/HomeXx/compuir/hands_ml/train.py"
    for line, ln in liveTraining(script):
        writeStats(idx, line, ln)

if __name__=="__main__":
    main()