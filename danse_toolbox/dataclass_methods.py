import ast
import yaml
import json, os
import numpy as np
import pickle, gzip
from copy import copy
from pathlib import Path
from io import TextIOWrapper
import dataclass_wizard as dcw
from dataclasses import fields, replace, make_dataclass, is_dataclass


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

    if hasattr(p, '__post_init__'):
        p.__post_init__()  # call the __post_init__ method if it exists

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


def dump_to_yaml_template(myDataclass, path=None):
    """Dumps a YAML template for a dataclass.
    
    Parameters
    ----------
    myDataclass : instance of a dataclass
        The dataclass to dump a template for.
    path : str
        The path to the YAML file to be created.
        If not provided, the file will be created in the current directory.
    """

    if path is None:
        path = f'{type(myDataclass).__name__}__template.yaml'

    def _convert_to_dict(cls):
        """
        Recursively converts a dataclass (and its nested dataclasses)
        to a dictionary.
        Difference with `dcw.asdict` is that this function does 
        not alter the capitalization of the field names, and keeps
        underscores in the field names.
        """
        outDict = {}
        for key in fields(cls):
            if is_dataclass(getattr(cls, key.name)):
                outDict[key.name] = _convert_to_dict(getattr(cls, key.name))
            elif type(getattr(cls, key.name)) is np.ndarray:
                # Convert numpy arrays to lists before dumping to YAML
                outDict[key.name] = getattr(cls, key.name).tolist()
            else:
                outDict[key.name] = getattr(cls, key.name)
        return outDict

    with open(path, 'w') as f:
        myDataclassAsDict = _convert_to_dict(myDataclass)
        yaml.dump(myDataclassAsDict, f, default_flow_style=False)
    
    print(f'YAML template for dataclass "{type(myDataclass).__name__}" dumped to "{path}".')


def load_from_yaml(path, myDataclass):
    """Loads data from a YAML file into a dataclass.
    
    Parameters
    ----------
    path : str
        The path to the YAML file to be loaded.
    myDataclass : instance of a dataclass
        The dataclass to load the data into.

    Returns
    -------
    myDataclass : instance of a dataclass
        The dataclass with the data loaded into it.
    """

    with open(path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    def _interpret_lists(d):
        """Interprets lists in the YAML file as lists of floats, not strings"""
        for key in d:
            if type(d[key]) is str and len(d[key]) >= 2:
                if d[key][0] == '[' and d[key][-1] == ']':
                    d[key] = ast.literal_eval(d[key])  # Convert string to list
                    # Use of `literal_eval` hinted at by https://stackoverflow.com/a/1894296
            elif type(d[key]) is dict:
                d[key] = _interpret_lists(d[key])
        return d

    # Detect lists
    d = _interpret_lists(d)

    def _deal_with_arrays(d, refDataclass):
        """Transforms lists that should be numpy arrays into numpy arrays"""
        for key in d:
            if type(d[key]) is list:
                refDict = {field.name: field.type for field in fields(refDataclass)}
                if refDict[key] is np.ndarray:
                    d[key] = np.array(d[key])
            elif type(d[key]) is dict:
                d[key] = _deal_with_arrays(d[key], getattr(refDataclass, key))
        return d

    # Deal with expected numpy arrays
    d = _deal_with_arrays(d, myDataclass)

    def _load_into_dataclass(d, myDataclass):
        """Loads data from a dict into a dataclass"""
        for key in d:
            if type(d[key]) is dict:
                setattr(
                    myDataclass,
                    key,
                    _load_into_dataclass(d[key], getattr(myDataclass, key))
                )
            else:
                setattr(myDataclass, key, d[key])
        return myDataclass

    # myDataclass = dcw.fromdict(myDataclass, d)
    myDataclass = _load_into_dataclass(d, myDataclass)

    # Re-initialize dataclass
    if hasattr(myDataclass, '__post_init__'):
        myDataclass.__post_init__()
    # Re-initialize sub-dataclasses
    for k, v in myDataclass.__dict__.items():
        if is_dataclass(v) and hasattr(v, '__post_init__'):
            v.__post_init__()

    return myDataclass


def dataclasses_equal(d1, d2):
    """
    Assesses whether two dataclasses (and their nested dataclasses) are equal.
    """
    for k, v in d1.__dict__.items():
        if is_dataclass(v):
            if not dataclasses_equal(v, getattr(d2, k)):
                return False
        elif isinstance(d2.__dict__[k], np.ndarray):
            if not d2.__dict__[k].shape == v.shape:
                return False
            if not np.allclose(d2.__dict__[k], v):
                return False
        elif d2.__dict__[k] != v:
            return False
    return True