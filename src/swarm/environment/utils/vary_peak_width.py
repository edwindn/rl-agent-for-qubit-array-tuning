import numpy as np

class VaryPeakWidth:
    def __init__(self, peak_width_0, alpha):
        self.peak_width_0 = peak_width_0
        self.alpha = alpha

    def linearly_vary_peak_width(self, v_x, v_y) -> float:
        v_avg = (abs(v_x) + abs(v_y)) / 2
        return self.peak_width_0 - self.alpha * v_avg
