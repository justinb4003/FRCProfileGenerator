# FRC Profile Generator from Trisonics 4003

## Installation

### Clone the repository
You can do this either by downloading the zip from GitHub and expanding it somewhere on your computer or, if you have Git Bash installed:
```
git clone https://github.com/justinb4003/FRCProfileGenerator.git
```

### Create and activate a virtual environment
We'll need Python 3 for this but we'll assume your python executable is just called `python` for this:
```
python -m venv venv-progen
```
Activate on Windows:
```
.\venv-progen\scripts\activate
```
Activate on macOS/Linux:
```
. ./venv-progen/bin/activate
```

### Install required packages 
This will install every package listed in the requirements.txt file. The widget library used here in wxWidgets, which is going to be the largest one to install. It's a nice abstration layer over native GUI controls for building applications. On Windows and macOS it's fairly simple to install as the native GUI controls are well defined. Linux doesn't have a single answer to "native controls" so the install there can be difficult depending on the distribution.
```
pip install -r requirements.txt
```

## Launch Application
```
python main.py
```
![Application screenshot showing Charged Up game field in one panel with control elements in another.](https://github.com/justinb4003/FRCProfileGenerator/assets/16728804/a4c3dc45-dea7-4e7e-bbc4-7c758e79731b)
