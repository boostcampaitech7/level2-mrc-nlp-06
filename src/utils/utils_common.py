import yaml
from box import Box

def configurer(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
        config = Box(config)

    return config