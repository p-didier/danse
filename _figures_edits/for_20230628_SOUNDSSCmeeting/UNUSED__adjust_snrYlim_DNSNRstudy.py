# Purpose of script:
# Adjust the ylim of barplots in the folders containing the results of the
# DANSE diffuse noise SNR study.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys

PATH_TO_FOLDERS = [f'danse/out/20230601_tests/{s}' for s in [
    'dn_effect',
    'dn_effect_gevd',
    'dn_effect_batch',
    'dn_effect_gevd_batch',
]]

def main(pathToFolders=PATH_TO_FOLDERS):
    """Main function (called by default when running script)."""
    data = read_data()

if __name__ == '__main__':
    sys.exit(main())