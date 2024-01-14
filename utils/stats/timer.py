import time


class Timer():
    def __init__(self) -> None:
        self.start_time = time.time()
        self.end_time = None
        return
    
    def reset(self) -> None:
        self.start_time = time.time()
        self.end_time = None
        return
    
    def end(self, desc:str=None) -> float:
        assert self.end_time is None, 'The end time is not None!'
        self.end_time = time.time()
        T = self.get_time()
        if desc is not None:
            print(desc)
        print('Time consumption: %.5f seconds'%(T))
        return T
    
    def get_time(self) -> float:
        assert self.start_time is not None and self.end_time is not None, 'The start and end times must not be None!'
        return self.end_time - self.start_time