import argparse
from time import time
import numpy as np

from Tweaker.MeshTweaker import Tweak
from Tweaker import FileHandler


def evaluate_tweaker(chromosome=None):
    extended_mode = True
    verbose = False
    show_progress = False
    convert = False
    favside = None
    volume = False
    inputfile = "Tweaker/demo_object.stl"
    #TODO iterate through multiple objects and compare to real values

    if chromosome is None:
        parameter = dict({"VECTOR_TOL": 0.001, "PLAFOND_ADV": 0.2, "FIRST_LAY_H": 0.25, "NEGL_FACE_SIZE": 1,
                          "ABSOLUTE_F": 100, "RELATIVE_F": 1, "CONTOUR_F": 0.5})
    else:
        parameter = dict()
        chromosome_names = [("VECTOR_TOL", 0.001), ("PLAFOND_ADV", 0.2), ("FIRST_LAY_H", 0.25), ("NEGL_FACE_SIZE", 1),
                     ("ABSOLUTE_F", 100), ("RELATIVE_F", 1), ("CONTOUR_F", 0.5)]
        for i, kv in enumerate(chromosome_names):
            parameter[kv[0]] = map_parameters(kv[0], chromosome[i])

    if verbose:
        print("Evaluating with parameter:")
        for key, val in parameter.items():
            print(f"  {key}:    \t{val}")

    filehandler = FileHandler.FileHandler()
    objs = filehandler.load_mesh(inputfile)
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

    # for best in x.best_5:
    #     print("  Aligment: [%-8s, %-8s, %-8s], Unprintability: %-10s "
    #           % (best[0][0].round(4), best[0][1].round(4), best[0][2].round(4), best[4].round(3)))

    # Transform the raw fitnesses to a single fitnes value
    # Return negative slope for negative and therefore invalid scores
    if x.unprintability < 0:
        return 1 + 10 * abs(x.unprintability),
    # logit transformation with a turning point in (10, 0.5) and a low value for x=0
    return 1/(1 + np.exp(0.5*(10-x.unprintability))),  # Return a tuple


def map_parameters(name, allele):
    """
    This functions maps the raw allele that is around 1 into an appropriate scale. Therefore, it maps the genotype
    onto the phenotype
    :param name: name of the gene
    :param allele: value for the specified gene
    :return: value that can be used by the algorithm
    """
    parameter = dict({"VECTOR_TOL": 0.001, "PLAFOND_ADV": 1, "FIRST_LAY_H": 1, "NEGL_FACE_SIZE": 1,
                      "ABSOLUTE_F": 100, "RELATIVE_F": 1, "CONTOUR_F": 1})
    return parameter[name] * allele

def map_all_parameters(chromosome):
    """
    This functions maps each allel of the chromosome that is around 1 into the appropriate scales.
    Therefore, it maps the genotype to the phenotype.
    :param chromosome: chromosome that contains all genes
    :return: list of the real values
    """
    chromosome_names = [("VECTOR_TOL", 0.001), ("PLAFOND_ADV", 1), ("FIRST_LAY_H", 1), ("NEGL_FACE_SIZE", 1),
                        ("ABSOLUTE_F", 100), ("RELATIVE_F", 1), ("CONTOUR_F", 1)]
    return [round(chromosome[i] * allele[1], 6) for i, allele in enumerate(chromosome_names)]


if __name__ == "__main__":
    evaluate_tweaker()
