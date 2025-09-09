# Mini Fire Detector 

A small PyTorch demo that trains a simple CNN to tell **fire vs. non-fire images**.  
For speed, it uses CIFAR-10 as a placeholder dataset — swap in a real fire dataset if you want something practical.

## What’s inside
- Tiny CNN model built with PyTorch
- One-epoch training loop
- Quick accuracy check on test data

## Try it yourself
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fire_detector.py
