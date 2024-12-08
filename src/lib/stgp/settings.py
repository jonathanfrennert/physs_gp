"""Global settings and setters/getters for gpjax."""

import bibtexparser
import pathlib
from enum import Enum

# useful flag for debugging
debug_mode = False
verbose = False

in_strict_mode = False

experimental_simple_time_weight = False
experimental_cumsum_time_weight = False
experimental_precomputed_segements = None
experimental_precomputed_num_segements = None
experimental_precomputed_cumsum_eps = 0.01

experimental_allow_f_multi_dim_per_output = False

# controls whether batchjax can convert loops into vmaps
use_loop_mode = False

# controls whether closed from ELL computations can be used
force_black_box = False

safe_mode = False

# controls whether any integral approximation should use monte-carlo or quadrature
use_quadrature = False

# when true will wrap filter P_k with force_symmetric
kalman_filter_force_symmetric = False

# useful for debugging
cvi_ng_exploit_space_time = True

cvi_ng_batch = True

# ===== Solver Specific settings ====
class SolveType(Enum):
    CHOLESKY = 1 # jax.scipy.linalg.cholesky_solve
    CG = 2 # jax.scipy.linalg.cg
    EXACT = 3 #  jax.scipy.linalg.solve

# controls what type of solve is preferred
# note this does not ensure that ALL solves will use this
# as some models/computations are coded only for a specific way (ie cholesky solves)
linear_solver: SolveType = SolveType.CHOLESKY

# parallel kalman filter
# when true will use the linear_solver for ALL solves
#   this is required as in the filtering operator we have found
#   that using linalg.solve is more stable
parallel_kf_force_linear_solve = False

# conjugate gradient settings
cg_precondition_rank = 20
cg_max_iter = 1000

whiten_space = False

jitter = 1e-5
ng_jitter = 1e-7
ng_samples = 10
ng_f_samples = 10
avoid_s_cholesky=False

class strict_mode:
    """Enable strict_mode.

    Use case:
        with settings.strict_mode():
            ...
    """

    def __init__(self, state=True, num_probe_vectors=1):
        """Store strict_mode state before with statement."""
        global in_strict_mode
        self.orig_value = in_strict_mode

    def __enter__(self):
        """Set strict_mode state to true."""
        global in_strict_mode
        in_strict_mode = True

    def __exit__(self, *args):
        """Restore strict_mode state."""
        global in_strict_mode
        in_strict_mode = self.orig_value

class use_loops:
    """Use loops instead of batching.

    Use case:
        with settings.use_loops():
            ...
    """

    def __init__(self, state=True, num_probe_vectors=1):
        """Store strict_mode state before with statement."""
        global use_loop_mode
        self.orig_value = use_loop_mode

    def __enter__(self):
        """Set strict_mode state to true."""
        global use_loop_mode
        use_loop_mode = True

    def __exit__(self, *args):
        """Restore strict_mode state."""
        global use_loop_mode
        use_loop_mode = self.orig_value


# global list of who to cite for the current model created
to_cite = []


def add_citation(arg):
    """Append citation to global to_cite list."""
    global to_cite
    to_cite.append(arg)


def print_citations():
    """Print bibtex citations."""
    global to_cite

    # this files path
    root = pathlib.Path(__file__).parent.absolute()

    # references are stored within gpjax, so use the root to this file to load
    with open(f"{root}/references/references.bib") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    bib_entries = bib_database.entries

    # TODO: print print these somehow
    for _id in to_cite:
        for entry in bib_entries:
            if _id == entry["ID"]:
                print(entry)
