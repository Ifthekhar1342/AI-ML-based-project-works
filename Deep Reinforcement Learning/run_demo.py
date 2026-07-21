
"""Small demonstration that does not require Stable-Baselines3."""

from pbf_rl_toolpath import (
    GeometricPBFEnv,
    ThermalPBFEnv,
    run_zigzag_episode,
    plot_episode,
)

import matplotlib.pyplot as plt


def main():
    geometric = GeometricPBFEnv(
        n=10,
        split="test",
        seed=1,
        fixed_mask_name="triangle",
    )
    print("Geometric zigzag:", run_zigzag_episode(geometric))
    plot_episode(geometric, "Geometric zigzag baseline")

    thermal = ThermalPBFEnv(
        n=10,
        split="test",
        seed=2,
        fixed_mask_name="ring",
    )
    print("Thermal zigzag:", run_zigzag_episode(thermal))
    plot_episode(thermal, "Thermal zigzag baseline")

    plt.show()


if __name__ == "__main__":
    main()
