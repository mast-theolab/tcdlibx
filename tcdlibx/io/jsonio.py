import os
import json

# Generic Functions
def write_json(dictionar: dict, fname: str) -> None:
    """Writes a file with a template JSON file with the Gaussian Options

    """
    towrite = json.dumps(dictionar, indent=4)
    with open(fname, 'w', encoding='utf-8') as fopen:
        fopen.write(towrite)


def read_json(fname: str) -> dict:
    """
    Read the option file. expected in json style
    """
    if not os.path.exists(fname):
        raise FileNotFoundError('Missing JSON file')
    with open(fname, 'r', encoding='UTF-8') as fopen:
        data = json.load(fopen)
    return data
