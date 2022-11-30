import copy
import pickle, gzip
from pathlib import Path
from dataclasses import fields, replace, make_dataclass, is_dataclass
from pathlib import PurePath
import json, os
import dataclass_wizard as dcw
import numpy as np
from io import TextIOWrapper


def save(self, foldername: str, exportType='pkl'):
    """
    Saves program settings so they can be loaded again later
    
    Parameters
    ----------
    self : dataclass
        Dataclass to be exported.
    foldername : str
        Folder where to export the dataclass.
    exportType : str
        Type of export. "json": exporting to JSON file. "pkl": exporting to PKL.GZ archive.
    """

    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        Path(foldername).mkdir(parents=True)
        print(f'Created output directory ".../{shortPath}".')

    fullPath = f'{foldername}/{type(self).__name__}'
    if exportType == 'pkl':
        fullPath += '.pkl.gz'
        pickle.dump(self, gzip.open(fullPath, 'wb'))
    elif exportType == 'json':
        fullPath += '.json'
        save_to_json(self, fullPath)
    # Also save as readily readable .txt file
    save_as_txt(self, foldername)

    print(f'<{type(self).__name__}> object data exported to directory\n".../{shortPath}".')


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


def save_as_txt(mycls, foldername):
    """Saves dataclass to TXT file for quick view."""

    def _write_lines(datacl, f: TextIOWrapper, ntabs=0):
        """Recursive helper function."""
        tabbing = ' |' * ntabs
        f.write(f'{tabbing}>--------{type(datacl).__name__} class instance\n')
        flds = [(fld.name, getattr(datacl, fld.name))\
            for fld in fields(datacl)]
        for ii in range(len(flds)):
            if is_dataclass(flds[ii][1]):
                _write_lines(flds[ii][1], f, ntabs+1)
            else:
                string = f'{tabbing} - {flds[ii][0]} = {flds[ii][1]}\n'
                f.write(string)
        f.write(f'{tabbing}_______\n')

    filename = f'{foldername}/{type(mycls).__name__}_text.txt'
    f = open(filename, 'w')
    _write_lines(mycls, f)
    f.close()


def save_to_json(mycls, filename):
    """Saves dataclass to JSON file"""
    
    raise ValueError('NOT YET CORRECTLY IMPLEMENTED')

    mycls = copy.copy(mycls)  # make a copy to not alter the original DC

    # Check extension
    filename_alone, file_extension = os.path.splitext(filename)
    if file_extension != '.json':
        print(f'The filename ("{filename}") should be that of a JSON file. Modifying...')
        filename = filename_alone + '.json'
        print(f'Filename modified to "{filename}".')

    # Convert arrays to lists before export
    mycls = convert_arrays_to_lists(mycls)

    jsondict = dcw.asdict(mycls)  # https://stackoverflow.com/a/69059343

    with open(filename, 'w') as file_json:
        json.dump(jsondict, file_json, indent=4)
    file_json.close()


def convert_arrays_to_lists(mycls):
    """
    Converts all arrays in dataclass to lists,
    with added element to indicate they previously
    were arrays. Does it recursively to cover
    all potential nested dataclasses.
    """

    for field in fields(mycls):
        val = getattr(mycls, field.name)
        if is_dataclass(val):
            setattr(mycls, field.name, convert_arrays_to_lists(val))
        elif type(val) is np.ndarray:
            newVal = val.tolist() + ['!!NPARRAY']  # include a flag to re-conversion to np.ndarray when reading the JSON file
            para = {field.name: newVal}
            mycls = replace(mycls, **para)  # TODO: <-- does not work with the danseWindow field... For some reason
            stop = 1
        elif type(val) is list:
            # TODO: should be made recursive too (to account for, e.g., lists of lists of arrays)
            if any([(type(v) is np.ndarray) for v in val]):
                newVal = []
                for ii in range(len(val)):
                    if type(val[ii]) is np.ndarray:
                        newVal.append(val[ii].tolist() + ['!!NPARRAY'])
                    else:
                        newVal.append(val[ii])
                para = {field.name: newVal}
                mycls = replace(mycls, **para)

    return mycls


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

def shorten_path(file_path, length=3):
    """Splits `file_path` into separate parts, select the last 
    `length` elements and join them again
    -- from: https://stackoverflow.com/a/49758154
    """
    return Path(*Path(file_path).parts[-length:])