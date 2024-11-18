# Autopilot using Reinforcement learning

This repository is used to store an implementation of a autopilot as described in my [blog post](https://markus.co/blog/autopilot).

## Running

1. Load manually or install (via pip) `jsbgym`
2. Install all the [requirements](requirements.txt)

```
pip install -r requirements.txt
```

3. Add Flightgear to path as needed (sample in in [run.sh](run.sh))
4. Train the model for a wanted task

```
   ./run.sh train HeadingControlTask
```

5. Evaluate using Flightgear visualtions

```
   ./run.sh eval HeadingControlTask
```
