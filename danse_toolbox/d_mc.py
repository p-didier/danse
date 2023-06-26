# Purpose of script:
# Functions for Monte-Carlo runs for DANSE-related experiments.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

from dataclasses import dataclass
import danse_toolbox.dataclass_methods as met

@dataclass
class MCExperiment:
    """Class for Monte-Carlo experiment."""
    seed: int = 0               # Seed for random number generator
    nMCruns: int = None         # Number of Monte-Carlo runs
    baseConfigFile: str = None  # Base config file (YAML)
    
    def load_from_yaml(self, path) -> 'MCExperiment':
        """Loads dataclass from YAML file."""
        self.loadedFromYaml = True  # flag to indicate that the object was loaded from YAML
        self.originYaml = path  # path to YAML file
        out = met.load_from_yaml(path, self)
        if hasattr(out, '__post_init__'):
            out.__post_init__()
        return out
    
    # def generate_random_parameters(self, signalsType: str='random'):
    #     """
    #     Generate random parameters for a single Monte-Carlo run.
        
    #     Parameters:
    #     signalsType: str
    #         Type of signals to use.
    #         Values implemented so far: 'random' (signals are drawn from
    #         random distributions).
    #     """
        
        