import sys
from time import time

from Tweaker.MeshTweaker import Tweak
from Tweaker import FileHandler


def evaluate_tweaker(parameter, input_file, verbose=False):
    if verbose:
        print(f"file: {input_file}")

    extended_mode = True
    show_progress = False
    convert = False
    favside = None
    volume = False

    filehandler = FileHandler.FileHandler()
    objs = filehandler.load_mesh(input_file)
    if objs is None:
        sys.exit()

    c = 0
    info = dict()
    for part, content in objs.items():
        mesh = content["mesh"]
        info[part] = dict()
        if convert:
            info[part]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            try:
                cstime = time()
                # print(f"  tweak {inputfile}")
                x = Tweak(mesh, extended_mode, verbose, show_progress, favside, volume,
                          **parameter)
                info[part]["matrix"] = x.matrix
                info[part]["tweaker_stats"] = x
            except (KeyboardInterrupt, SystemExit):
                raise SystemExit("\nError, tweaking process failed!")

            # List tweaking results
            if verbose:
                print("Result-stats:")
                print(" Tweaked Z-axis: \t{}".format(x.alignment))
                print(" Axis, angle:   \t{}".format(x.euler_parameter))
                print(""" Rotation matrix:
            {:2f}\t{:2f}\t{:2f}
            {:2f}\t{:2f}\t{:2f}
            {:2f}\t{:2f}\t{:2f}""".format(x.matrix[0][0], x.matrix[0][1], x.matrix[0][2],
                                          x.matrix[1][0], x.matrix[1][1], x.matrix[1][2],
                                          x.matrix[2][0], x.matrix[2][1], x.matrix[2][2]))
                print(" Unprintability: \t{}".format(x.unprintability))

                print("Found result:    \t{:2f} s\n".format(time() - cstime))
    return x


if __name__ == "__main__":
    evaluate_tweaker()
