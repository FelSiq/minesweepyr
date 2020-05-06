# Minesweeper in Python
![Python minesweeper ('easy' difficult)](images/ms1.png)
This is a simple implementation of Minesweeper fully in Python.

<img alt="Python minesweeper ('medium' difficult)" src="images/ms2.png" width="960" />
<img alt="Python minesweeper ('expert' difficult)" src="images/ms3.png" width="960" />

# Requirements
Python 3.8+

# Installation
```
git clone https://github.com/FelSiq/minesweepyr.git
pip install -Ur requirements.txt
```

# Run
While running the python script, the first argument must be the difficult level ('easy', 'medium', or 'expert'). Additionally, you may send a numeric random seed as the second (optional) argument to repeat the same game sequences.
```
python minesweepyr.py (easy|medium|expert) [random_seed]
```

# Commands
Standard minesweeper commands:
- **Left click:** open tile;
- **Right click:** switch mark in tile; and
- **Both/middle button:** open all tiles in the neighborhood of target tile.

# License
MIT.
