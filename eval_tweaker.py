import argparse
from time import time

if __name__ == '__main__':
    from Tweaker.Tweaker import  getargs
    from Tweaker.MeshTweaker import Tweak
    from Tweaker import FileHandler
else:
    from .Tweaker.MeshTweaker import Tweak
    from .Tweaker import FileHandler
    from .Tweaker import FileHandler

args = getargs()
args.extended_mode = True
args.verbose = False
args.show_progress = False
args.convert = False
args.favside = None
args.volume = False
args.inputfile = "Tweaker/demo_object.stl"

# Constants used and to optimize
VECTOR_TOL = 0.001  # To remove alignment duplicates, the vector tolerance is
# used to distinguish two vectors.
PLAFOND_ADV = 0.2  # Printing a plafond is known to be more effective than
# very step overhangs. This value sets the advantage in %.
FIRST_LAY_H = 0.25  # The initial layer of a print has an altitude > 0
# bottom layer and very bottom-near overhangs can be handled as similar.
NEGL_FACE_SIZE = 1  # The fast operation mode neglects facet sizes smaller than
# this value (in mm^2) for a better performance
ABSOLUTE_F = 100  # These values scale the the parameters bottom size,
RELATIVE_F = 1  # overhang size, and bottom contour length to get a robust
CONTOUR_F = 0.5  # value for the Unprintability


FileHandler = FileHandler.FileHandler()
objs = FileHandler.load_mesh(args.inputfile)
if objs is None:
    sys.exit()

c = 0
info = dict()
for part, content in objs.items():
    mesh = content["mesh"]
    info[part] = dict()
    if args.convert:
        info[part]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        try:
            cstime = time()
            x = Tweak(mesh, args.extended_mode, args.verbose, args.show_progress, args.favside, args.volume)
            info[part]["matrix"] = x.matrix
            info[part]["tweaker_stats"] = x
        except (KeyboardInterrupt, SystemExit):
            raise SystemExit("\nError, tweaking process failed!")

        # List tweaking results
        if args.result or args.verbose:
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

for best in x.best_5:
    print("  Aligment: [%-8s, %-8s, %-8s], Unprintability: %-10s "
          % (best[0][0].round(4), best[0][1].round(4), best[0][2].round(4), best[4].round(3)))
