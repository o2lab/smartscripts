""" Inconsistent data dependency """


class Figure:
    """ Fake figure class """
    def __init__(self, name=''):
        self.name = name

    def other(self):
        """ another public func to eliminate R0903 check"""

    def add(self, layout):
        """ fake add_subplot function """
        return self.name + layout


def misuse(plot, with_components):
    """ Bug example """
    if plot is True:
        fig = Figure()
        if with_components:
            axis = fig.add(121)
            axis2 = fig.add(122)
        else:
            axis = fig.add(111)
    # ... some other statements
    if plot is True:
        print(axis)
        if with_components:
            print(axis2)
