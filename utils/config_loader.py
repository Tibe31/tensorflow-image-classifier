import yaml
import ast

class Config:
    def __init__(self, path='config.yaml'):
        with open(path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Robust conversion for input_shape in case it's read as a string
        if 'input_shape' in self.cfg and isinstance(self.cfg['input_shape'], str):
            try:
                self.cfg['input_shape'] = ast.literal_eval(self.cfg['input_shape'])
            except (ValueError, SyntaxError):
                # Handle error or log if conversion fails
                pass # Keep as string if conversion fails, or raise an error
        
        # Ensure it's a tuple of integers
        if 'input_shape' in self.cfg and isinstance(self.cfg['input_shape'], list):
            self.cfg['input_shape'] = tuple(self.cfg['input_shape'])
        elif 'input_shape' in self.cfg and isinstance(self.cfg['input_shape'], tuple):
            # Ensure all elements are integers
            self.cfg['input_shape'] = tuple(int(x) for x in self.cfg['input_shape'])

    def __getitem__(self, item):
        return self.cfg[item]

    def get(self, item, default=None):
        return self.cfg.get(item, default)