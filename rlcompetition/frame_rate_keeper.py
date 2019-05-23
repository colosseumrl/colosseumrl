import time


class FrameRateKeeper:
    def __init__(self, max_frame_rate: float):
        self.max_frame_rate: float = max_frame_rate
        self.max_frame_time: float = 1.0 / max_frame_rate
        self.frame_start_time: float = time.time()

    def tick(self):
        end_time = time.time()
        wait_time = self.max_frame_time - (end_time - self.frame_start_time)
        if wait_time > 0:
            time.sleep(wait_time)
        self.frame_start_time = time.time()
