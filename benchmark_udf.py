import argparse
import cProfile
import pstats
from pathlib import Path

import numpy as np

from utils.skel import udf_pixel


SAMPLE_TEXT = (
    "2018.035616438356 2983.0, 2018.0849315068492 3342.0, 2018.0986301369862 3106.0, "
    "2018.1041095890412 3160.0, 2018.1178082191782 3011.0, 2018.1178082191782 -9999, "
    "2018.13698630137 2731.0, 2018.145205479452 2857.0, 2018.1616438356164 2782.0, "
    "2018.1671232876713 2572.0, 2018.2054794520548 -9999, 2018.2082191780821 2436.0, "
    "2018.2246575342465 2881.0, 2018.227397260274 -9999, 2018.2493150684932 2825.0, "
    "2018.2493150684932 2890.0, 2018.2630136986302 -9999, 2018.268493150685 3965.0, "
    "2018.268493150685 3975.0, 2018.2904109589042 5382.0, 2018.2931506849316 5290.0, "
    "2018.295890410959 5898.0, 2018.304109589041 -9999, 2018.317808219178 -9999, "
    "2018.323287671233 7505.0, 2018.33698630137 7889.0, 2018.33698630137 8057.0, "
    "2018.345205479452 8228.0, 2018.3506849315067 8488.0, 2018.3643835616438 9036.0, "
    "2018.3780821917808 -9999, 2018.3808219178081 9042.0, 2018.386301369863 9182.0, "
    "2018.3917808219178 -9999, 2018.4 -9999, 2018.4054794520548 9255.0, "
    "2018.4136986301369 -9999, 2018.427397260274 8679.0, 2018.4328767123288 8533.0, "
    "2018.441095890411 8628.0, 2018.495890410959 5672.0, 2018.5013698630137 -9999, "
    "2018.5123287671233 5107.0, 2018.5287671232877 5261.0, 2018.531506849315 6430.0, "
    "2018.5369863013698 6234.0, 2018.5424657534247 6375.0, 2018.5506849315068 -9999, "
    "2018.5561643835617 -9999, 2018.5561643835617 -9999, 2018.5643835616438 6787.0, "
    "2018.5698630136985 7416.0, 2018.5753424657535 7059.0, 2018.5780821917808 7079.0, "
    "2018.5972602739726 7322.0, 2018.6 7888.0, 2018.6109589041096 -9999, "
    "2018.6383561643836 7313.0, 2018.6657534246576 -9999, 2018.6739726027397 -9999, "
    "2018.6794520547944 7208.0, 2018.6876712328767 5541.0, 2018.6876712328767 4451.0, "
    "2018.7150684931507 6746.0, 2018.731506849315 7893.0, 2018.7616438356165 2303.0, "
    "2018.7753424657535 3070.0, 2018.7753424657535 3107.0, 2018.7835616438356 3265.0, "
    "2018.7890410958903 3461.0, 2018.8027397260273 3743.0, 2018.8301369863013 -9999, "
    "2018.8438356164384 -9999, 2018.8794520547945 2259.0, 2018.9068493150685 2873.0, "
    "2018.9068493150685 2686.0, 2018.9260273972602 2832.0, 2018.9260273972602 2874.0"
)


def parse_sample_series():
    values = np.array(SAMPLE_TEXT.replace(", ", " ").split(" "), dtype=float).reshape(-1, 2)
    x = values[:, 0]
    y = values[:, 1]
    return x, y


def benchmark_detect(iterations):
    udf_pixel.GLstart = 0.2
    udf_pixel.GLend = 1
    udf_pixel.PSstart = 0.33
    udf_pixel.PSend = 0.66
    udf_pixel.GFstd = 0.02
    udf_pixel.posEval = 40
    udf_pixel.clrwd = 15
    udf_pixel.profileAnalytics = False

    x, y = parse_sample_series()
    for _ in range(iterations):
        udf_pixel.detectMow_S2_new(x, y, clearWd=15, yr=2018, type="ConHull", nOrder=3, model="linear")


def benchmark_forcepy(iterations):
    _, y = parse_sample_series()
    dates = np.arange(17545, 17545 + len(y), dtype=int)
    inarray = y.reshape(len(y), 1, 1, 1)
    outarray = np.zeros(17, dtype=float)
    for _ in range(iterations):
        udf_pixel.forcepy_pixel(inarray, outarray, dates, None, None, -9999, 1)


def profile_runner(func, iterations, profile_output):
    profiler = cProfile.Profile()
    profiler.enable()
    func(iterations)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)
    if profile_output:
        stats.dump_stats(profile_output)
        print(f"profile written to {profile_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["detect", "forcepy"], default="detect")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--profile-output", type=Path, default=None)
    args = parser.parse_args()

    runner = benchmark_detect if args.mode == "detect" else benchmark_forcepy
    profile_runner(runner, args.iterations, args.profile_output)


if __name__ == "__main__":
    main()
