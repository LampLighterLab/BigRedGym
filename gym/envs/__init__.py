import importlib

from gym.utils.task_registry import task_registry

# * To add a new env:
# * 1. add the base env and env class name and location to the class dict
# * 2. add the config name and location to the config dict
# * 3. add the runner confg name and location to the runner config dict
# * 3. register the task experiment name to the env/config/ppo classes

class_dict = {
    "LeggedRobot": ".base.legged_robot",
    "FixedRobot": ".base.fixed_robot",
    "Cartpole": ".cartpole.cartpole",
    "MiniCheetah": ".mini_cheetah.mini_cheetah",
    "MiniCheetahRef": ".mini_cheetah.mini_cheetah_ref",
    "MiniCheetahOsc": ".mini_cheetah.mini_cheetah_osc",
    "MIT_Humanoid": ".mit_humanoid.mit_humanoid",
    "HumanoidRunning": ".mit_humanoid.humanoid_running",
    "Pendulum": ".pendulum.pendulum",
    "Go2": ".go2.go2",
    "Go2Trot": ".go2.go2trot",
    "Go2SE": ".go2.go2_se",
    "Go2SEJump": ".go2.go2_se_jump",
    "Go2SEJumpReal": ".go2.go2_se_jump",
}

config_dict = {
    "CartpoleCfg": ".cartpole.cartpole_config",
    "MiniCheetahCfg": ".mini_cheetah.mini_cheetah_config",
    "MiniCheetahRefCfg": ".mini_cheetah.mini_cheetah_ref_config",
    "MiniCheetahRefVSimCfg": ".mini_cheetah.mini_cheetah_ref_vsim_config",
    "MiniCheetahOscCfg": ".mini_cheetah.mini_cheetah_osc_config",
    "MiniCheetahSACCfg": ".mini_cheetah.mini_cheetah_SAC_config",
    "MITHumanoidCfg": ".mit_humanoid.mit_humanoid_config",
    "HumanoidRunningCfg": ".mit_humanoid.humanoid_running_config",
    "PendulumCfg": ".pendulum.pendulum_config",
    "PendulumSACCfg": ".pendulum.pendulum_SAC_config",
    "PendulumPSDCfg": ".pendulum.pendulum_PSD_config",
    "Go2Cfg": ".go2.go2_config",
    "Go2TrotCfg": ".go2.go2trot_config",
    "Go2SECfg": ".go2.go2_se_config",
    "Go2SEPaperCfg": ".go2.go2_se_config",
    "Go2SEGo2EnvCfg": ".go2.go2_se_config",
    "Go2SEJumpCfg": ".go2.go2_se_config",
    "Go2SEJumpRealCfg": ".go2.go2_se_config",
}

runner_config_dict = {
    "CartpoleRunnerCfg": ".cartpole.cartpole_config",
    "MiniCheetahRunnerCfg": ".mini_cheetah.mini_cheetah_config",
    "MiniCheetahRefRunnerCfg": ".mini_cheetah.mini_cheetah_ref_config",
    "MiniCheetahRefVSimRunnerCfg": ".mini_cheetah.mini_cheetah_ref_vsim_config",
    "MiniCheetahOscRunnerCfg": ".mini_cheetah.mini_cheetah_osc_config",
    "MiniCheetahSACRunnerCfg": ".mini_cheetah.mini_cheetah_SAC_config",
    "MITHumanoidRunnerCfg": ".mit_humanoid.mit_humanoid_config",
    "HumanoidRunningRunnerCfg": ".mit_humanoid.humanoid_running_config",
    "PendulumRunnerCfg": ".pendulum.pendulum_config",
    "PendulumSACRunnerCfg": ".pendulum.pendulum_SAC_config",
    "PendulumPSDRunnerCfg": ".pendulum.pendulum_PSD_config",
    "Go2RunnerCfg": ".go2.go2_config",
    "Go2TrotRunnerCfg": ".go2.go2trot_config",
    "Go2SERunnerCfg": ".go2.go2_se_config",
    "Go2SEOracleRunnerCfg": ".go2.go2_se_config",
    "Go2SEPaperRunnerCfg": ".go2.go2_se_config",
    "Go2SEJumpRunnerCfg": ".go2.go2_se_config",
    "Go2SEJumpOracleRunnerCfg": ".go2.go2_se_config",
    "Go2SEJumpRealRunnerCfg": ".go2.go2_se_config",
    "Go2DiagARunnerCfg": ".go2.go2_se_config",
    "Go2DiagBRunnerCfg": ".go2.go2_se_config",
    "Go2DiagCRunnerCfg": ".go2.go2_se_config",
}

task_dict = {
    "cartpole": ["Cartpole", "CartpoleCfg", "CartpoleRunnerCfg"],
    "mini_cheetah": ["MiniCheetah", "MiniCheetahCfg", "MiniCheetahRunnerCfg"],
    "mini_cheetah_ref": [
        "MiniCheetahRef",
        "MiniCheetahRefCfg",
        "MiniCheetahRefRunnerCfg",
    ],
    "mini_cheetah_ref_vsim": [
        "MiniCheetahRef",
        "MiniCheetahRefVSimCfg",
        "MiniCheetahRefVSimRunnerCfg",
    ],
    "mini_cheetah_osc": [
        "MiniCheetahOsc",
        "MiniCheetahOscCfg",
        "MiniCheetahOscRunnerCfg",
    ],
    "sac_mini_cheetah": [
        "MiniCheetahRef",
        "MiniCheetahSACCfg",
        "MiniCheetahSACRunnerCfg",
    ],
    "humanoid": ["MIT_Humanoid", "MITHumanoidCfg", "MITHumanoidRunnerCfg"],
    "humanoid_running": [
        "HumanoidRunning",
        "HumanoidRunningCfg",
        "HumanoidRunningRunnerCfg",
    ],
    "pendulum": ["Pendulum", "PendulumCfg", "PendulumRunnerCfg"],
    "sac_pendulum": ["Pendulum", "PendulumSACCfg", "PendulumSACRunnerCfg"],
    "psd_pendulum": ["Pendulum", "PendulumPSDCfg", "PendulumPSDRunnerCfg"],
    "go2": ["Go2", "Go2Cfg", "Go2RunnerCfg"],
    "go2trot": ["Go2Trot", "Go2TrotCfg", "Go2TrotRunnerCfg"],
    "go2_se": ["Go2SE", "Go2SECfg", "Go2SERunnerCfg"],
    "go2_se_oracle": ["Go2SE", "Go2SECfg", "Go2SEOracleRunnerCfg"],
    "go2_se_paper": ["Go2SE", "Go2SEPaperCfg", "Go2SEPaperRunnerCfg"],
    "go2_se_jump": ["Go2SEJump", "Go2SEJumpCfg", "Go2SEJumpRunnerCfg"],
    "go2_se_jump_oracle": ["Go2SEJump", "Go2SEJumpCfg", "Go2SEJumpOracleRunnerCfg"],
    "go2_se_jump_real": ["Go2SEJumpReal", "Go2SEJumpRealCfg", "Go2SEJumpRealRunnerCfg"],
    "go2_se_go2env": ["Go2SE", "Go2SEGo2EnvCfg", "Go2SERunnerCfg"],
    "diag_a": ["Go2SE", "Go2SEGo2EnvCfg", "Go2DiagARunnerCfg"],
    "diag_b": ["Go2SE", "Go2SEGo2EnvCfg", "Go2DiagBRunnerCfg"],
    "diag_c": ["Go2SE", "Go2SEGo2EnvCfg", "Go2DiagCRunnerCfg"],
}


def _load_declared_symbols(symbol_locations):
    loaded = {}
    for symbol, module in symbol_locations.items():
        loaded[symbol] = getattr(importlib.import_module(module, __name__), symbol)
        globals()[symbol] = loaded[symbol]
    return loaded


# Every declaration is required. Import and attribute failures intentionally
# abort registration instead of making a broken task disappear from the CLI.
_loaded = {}
for declarations in (class_dict, config_dict, runner_config_dict):
    _loaded.update(_load_declared_symbols(declarations))

for task_name, (class_name, config_name, runner_config_name) in task_dict.items():
    task_registry.register(
        task_name,
        _loaded[class_name],
        _loaded[config_name](),
        _loaded[runner_config_name](),
    )
