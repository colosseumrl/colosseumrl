from time import time, sleep


class FrameRateKeeper:
    """ Class for maintaining frame rate on server and keeping track of timeouts. """
    def __init__(self, max_frame_rate: float):
        self.max_frame_rate: float = max_frame_rate
        self.max_frame_time: float = 1.0 / max_frame_rate
        self.frame_start_time: float = time()

        self.timeout_start_time: float = time()
        self.timeout_time: float = float('inf')

    def tick(self) -> bool:
        """ Wait for a single frame roll to finish before returning and check to see if we have timed out

        Returns
        -------
        bool: Whether or not we have timed out
        """

        # Wait until the next frame roll
        wait_time = self.max_frame_time - (time() - self.frame_start_time)
        if wait_time > 0:
            sleep(wait_time)

        # Update time
        self.frame_start_time = time()

        # See if the timeout has triggered
        return (self.frame_start_time - self.timeout_start_time) > self.timeout_time

    def start_timeout(self, seconds: float):
        """ Initiate a new timeout. You can only have one active timeout at a time.

        Parameters
        ----------
        seconds: Number of seconds before triggering

        """
        self.timeout_start_time: float = time()
        self.timeout_time = seconds
