# utils/earlystopping.py

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min', enabled=True):
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.enabled = enabled
        
        self.best = float('inf') if self.mode =='min' else -float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if not self.enabled:
            return False
        
        if self.mode == 'min':
            if value < self.best - self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value > self.best + self.min_delta:
                self.best = value
                self.counter = 0
            else:
                self.counter += 1

        self.should_stop = (self.counter >= self.patience)
        return self.should_stop