# Mini Fire Detector 🔥

This is a **toy project** showing how to train a simple CNN for fire vs. non-fire detection.  
Right now it runs on CIFAR-10 (because it's small and built-in), but you can swap in a real fire dataset later.

## Why this repo exists
I wanted a quick, hands-on demo to show:
- How to build a CNN in PyTorch
- How to train/test on image data
- How to structure a clean ML script

## What it does
- Downloads CIFAR-10
- Trains a small CNN for one epoch
- Prints training loss and test accuracy

## Quick start
```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run training
python fire_detector.py
```

## Next steps
- Replace CIFAR-10 with a fire dataset
- Balance classes (fire / not fire)
- Save trained weights and deploy in a real system

---

MIT License
