import yaml

class Config:
    def __init__(self, path='config.yaml'):
        with open(path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.cfg[item]

    def get(self, item, default=None):
        return self.cfg.get(item, default)