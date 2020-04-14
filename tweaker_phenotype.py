import sys
from time import time

from Tweaker.MeshTweaker import Tweak
from Tweaker import FileHandler

# chromosome_mapping = [("VECTOR_TOL", 0.00132), ("PLAFOND_ADV", 0.1035), ("FIRST_LAY_H", 0.1055),
                      # ("NEGL_FACE_SIZE", 2.44), ("ABSOLUTE_F", 340.0), ("RELATIVE_F", 0.1109), ("CONTOUR_F", 0.52)]
# (2.1221719828604946, 1.75)
chromosome_mapping = [("VECTOR_TOL", 0.001), ("PLAFOND_ADV", 1.0), ("FIRST_LAY_H", 0.1),
                      ("NEGL_FACE_SIZE", 1.0), ("ABSOLUTE_F", 100.0), ("RELATIVE_F", 1.0), ("CONTOUR_F", 1.0)]
# Orig: [0.001, 0.2, 0.25, 1, 100, 1, 0.5] (4.052568577348239, 3.25)
# 2020-04-14: [0.001241, 0.220733, 0.027727, 1.396286, 222.797949, 0.368569, 0.145722] (1.8986392001671926, 1.5)

chromosome_dict = dict(chromosome_mapping)


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


def map_parameters(name, allele):
    """
    This functions maps the raw allele that is around 1 into an appropriate scale. Therefore, it maps the genotype
    onto the phenotype
    :param name: name of the gene
    :param allele: value for the specified gene
    :return: value that can be used by the algorithm
    """
    return chromosome_dict[name] * allele


def map_all_parameters(chromosome):
    """
    This functions maps each allel of the chromosome that is around 1 into the appropriate scales.
    Therefore, it maps the genotype to the phenotype.
    :param chromosome: chromosome that contains all genes
    :return: list of the real values
    """
    return [round(chromosome[i] * allele[1], 6) for i, allele in enumerate(chromosome_mapping)]


if __name__ == "__main__":
    evaluate_tweaker()
