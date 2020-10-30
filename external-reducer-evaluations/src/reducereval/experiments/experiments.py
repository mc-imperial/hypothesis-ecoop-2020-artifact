import attr


@attr.s()
class Experiment(object):
    generator = attr.ib()
    calculate_info = attr.ib()
    calculate_error_predicate = attr.ib()
    normalize_test_case = attr.ib()
    check_validity = attr.ib()

    name = attr.ib()
    external_reduction = attr.ib(default=True)


EXPERIMENTS = {}


def define_experiment(name, *args, **kwargs):
    assert name not in EXPERIMENTS
    kwargs["name"] = name
    result = Experiment(*args, **kwargs)
    EXPERIMENTS[name] = result
