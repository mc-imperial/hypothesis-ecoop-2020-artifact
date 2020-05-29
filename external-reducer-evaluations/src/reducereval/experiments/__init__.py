from reducereval.experiments.experiments import EXPERIMENTS, Experiment

import reducereval.experiments.csmith  # noqa
import reducereval.experiments.formatting  # noqa

assert set(EXPERIMENTS) == {"csmith", "formatting"}
