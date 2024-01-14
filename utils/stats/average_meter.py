'''
@Author: MHC
@Desc: General meter for getting average value.
'''
class AverageMeter():
    def __init__(self, mode:str='min') -> None:
        super(AverageMeter, self).__init__()
        assert mode in ['min', 'max'], 'Mode is either min or max.'
        self.mode = mode
        self.counter = 0
        self.sum = 0.
        self.avg = 0.
        self.best = None
    
    def update(self, val) -> None:
        self.counter += 1
        self.sum += val
        return
    
    def get_avg(self) -> float:
        self.avg = self.sum / self.counter
        if self.best == None:
            self.best = self.avg
        else:
            if self.mode == 'min':
                if self.avg < self.best:
                    self.best = self.avg
            else:
                if self.avg > self.best:
                    self.best = self.avg
        return self.avg
    
    def get_best(self) -> float:
        return self.best
    
    def reset(self, all:bool=False) -> None:
        self.counter = 0
        self.sum = 0.
        self.avg = 0.
        if all:
            self.best = None
        return
    