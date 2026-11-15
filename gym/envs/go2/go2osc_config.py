import math

from gym.envs.go2.go2_config import Go2Cfg

class Go2OscCfg(Go2Cfg):
    class osc:
        omega = 2.0
        sigma = 1.0
        xi = 0.0
        amplitude = math.pi / 4