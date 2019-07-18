# Linear Roughness

Linear Roughness is a Python library for extracting the position of a linear domain wall from an image, calculating various correlation functions on the position data and extracting roughness parameters from these functions.
This library was developed as part of my Master's Thesis project in Condensed Matter Physics at the Instituto Balseiro. It is written in Python 2.7 and it was intended to be used with Jupyter Notebooks in the laboratory or while analyzing numerical simulations.
Future changes will include porting to Python 3 and being able to use it without a Jupyter Notebook.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Linear Roughness from local files:

```bash
cd /path/to/setup.py
pip install linear_roughness .
```

## Usage
Use the included Jupyter Notebook which includes all the necessary functions to work with.
```python
from linear_roughness import *
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
