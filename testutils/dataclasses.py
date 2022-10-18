
import json, os
import numpy as np
import pickle, gzip
from copy import copy
from pathlib import Path
import dataclass_wizard as dcw
from dataclasses import fields, make_dataclass

def load(self, foldername: str, silent=False, dataType='pkl'):
    """
    Loads program settings object from file

    Parameters
    ----------
    self : dataclass
        Dataclass to be exported.
    foldername : str
        Folder where to export the dataclass.
    silent : bool
        If True, no printouts.
    dataType : str
        Type of file to import. "json": JSON file. "pkl": PKL.GZ archive.
    """
    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        raise ValueError(f'The folder "{foldername}" cannot be found.')

    if dataType == 'pkl':
        baseExtension = '.pkl.gz'
        altExtension = '.json'
    elif dataType == 'json':
        baseExtension = '.json'
        altExtension = '.pkl.gz'
    
    pathToFile = f'{foldername}/{type(self).__name__}{baseExtension}'
    if not Path(pathToFile).is_file():
        pathToAlternativeFile = f'{foldername}/{type(self).__name__}{altExtension}'
        if Path(pathToAlternativeFile).is_file():
            print(f'The file\n"{pathToFile}"\ndoes not exist. Loading\n"{pathToAlternativeFile}"\ninstead.')
            pathToFile = copy(pathToAlternativeFile)
            baseExtension = copy(altExtension)
        else:
            raise ValueError(f'Import issue, file\n"{pathToFile}"\nnot found (with either possible extensions).')

    if baseExtension == '.json':
        p = load_from_json(pathToFile, self)
    elif baseExtension == '.pkl.gz':
        p = pickle.load(gzip.open(pathToFile, 'r'))
    else:
        raise ValueError(f'Incorrect base extension: "{baseExtension}".')
    
    if not silent:
        print(f'<{type(self).__name__}> object data loaded from directory\n".../{shortPath}".')

    return p

    
def shorten_path(file_path, length=3):
    """Splits `file_path` into separate parts, select the last 
    `length` elements and join them again
    -- from: https://stackoverflow.com/a/49758154
    """
    return Path(*Path(file_path).parts[-length:])


def load_from_json(path_to_json_file, mycls):
    """Loads dataclass from JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(path_to_json_file)
    if file_extension != '.json':
        print(f'The filename ("{path_to_json_file}") should be that of a JSON file. Modifying...')
        path_to_json_file = filename_alone + '.json'
        print(f'Filename modified to "{path_to_json_file}".')

    with open(path_to_json_file) as fp:
        d = json.load(fp)

    # Create surrogate class
    c = make_dataclass('MySurrogate', [(key, type(d[key])) for key in d])
    # Fill it in with dict entries <-- TODO -- probably simpler to directly fill in `mycls_out` from `d`
    mycls_surrog = dcw.fromdict(c, d)

    # Fill in a new correct instance of the desired dataclass
    mycls_out = copy.copy(mycls)
    for field in fields(mycls_surrog):
        a = getattr(mycls_surrog, field.name)
        if field.type is list:
            if a[-1] == '!!NPARRAY':
                a = np.array(a[:-1])
            else: # TODO: should be made recursive (see other TODO in "save" function above)
                for ii in range(len(a)):
                    if a[ii][-1] == '!!NPARRAY':
                        a[ii] = np.array(a[ii][:-1])
        setattr(mycls_out, field.name, a)

    return mycls_out
