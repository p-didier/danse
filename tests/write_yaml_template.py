# Purpose of script:
# Writes a YAML template for a dataclass.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from pathlib import Path
from danse_toolbox.dataclass_methods import dump_to_yaml_template
from danse_toolbox.d_classes import TestParameters

DATACLASS_NAME = TestParameters
BASE_PATH_EXPORT = f'{Path(__file__).parent.parent}/config_files'

def main():
    """Main function (called by default when running script)."""
    
    # Write YAML template
    instance = DATACLASS_NAME()
    dump_to_yaml_template(
        instance,
        path=f'{BASE_PATH_EXPORT}/{DATACLASS_NAME.__name__}_cfg.yaml'
    )

if __name__ == '__main__':
    sys.exit(main())