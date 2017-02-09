from functools import partial

from elfi.v2.utils import scipy_from_str
from elfi.v2.network import NodeReference
from elfi.v2.operations import rvs_operation


class RandomVariable(NodeReference):
    def __init__(self, name, distribution="uniform", *params, size=None, **kwargs):
        state = dict(distribution=distribution, size=size)
        super(RandomVariable, self).__init__(name, *params, state=state, **kwargs)

    @staticmethod
    def compile(state):
        size = state['size']
        distribution = state['distribution']
        if not (size is None or isinstance(size, tuple)):
            size = (size, )

        if isinstance(distribution, str):
            distribution = scipy_from_str(distribution)

        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} "
                             "must implement a rvs method".format(distribution))

        output = partial(rvs_operation, distribution=distribution, size=size)
        return dict(output=output, random_state=True)

    def __str__(self):
        d = self['distribution']

        if isinstance(d, str):
            name = "'{}'".format(d)
        elif hasattr(d, 'name'):
            name = "'{}'".format(d.name)
        elif isinstance(d, type):
            name = d.__name__
        else:
            name = d.__class__.__name__

        return super(RandomVariable, self).__str__()[0:-1] + ", {})".format(name)


class Prior(RandomVariable):
    pass


class Simulator(NodeReference):
    def __init__(self, name, fn, *dependencies, observed=None, **kwargs):
        state = dict(fn=fn, observed=observed)
        super(Simulator, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile(state):
        fn = state['fn']
        return dict(output=fn, random_state=True)


class Summary(NodeReference):
    def __init__(self, name, fn, *dependencies, observed=None, **kwargs):
        state = dict(fn=fn, observed=observed)
        super(Summary, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile(state):
        fn = state['fn']
        return dict(output=fn)


class Discrepancy(NodeReference):
    def __init__(self, name, fn, *dependencies, **kwargs):
        state = dict(fn=fn)
        super(Discrepancy, self).__init__(name, *dependencies, state=state, **kwargs)

    @staticmethod
    def compile(state):
        fn = state['fn']
        return dict(output=fn)



















