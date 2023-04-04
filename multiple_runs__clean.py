# Purpose of script:
# Running proof-of-concept tests to show the effect of SROs on the functioning
# of TI-DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import yaml
from pathlib import Path

PATH_TO_CONFIG_FILE = f'{Path(__file__).parent}/config_files/sros_effect.yaml'

def main():
    """Main function (called by default when running script)."""
    cfg = read_config(filePath=PATH_TO_CONFIG_FILE)

    # Run tests
    run_test_batch(cfg)


def read_config(filePath: str):
    """
    Reads the YAML configuration file.
    
    Parameters
    ----------
    filePath : str
        Path to YAML configuration file.
    
    Returns
    ----------
    cfg : dict
        Configuration object.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    with open(filePath, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    return cfg


def run_test_batch(cfg: dict):
    """
    Runs a test batch based on a (YAML) configuration.
    
    Parameters
    ----------
    cfg : dict
        Configuration object.
    
    Returns
    ----------
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    srosToConsider = cfg['sros']


if __name__ == '__main__':
    sys.exit(main())