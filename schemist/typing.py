"""Types used in schemist."""

from collections import namedtuple

DataSplits = namedtuple('DataSplits',
                        ['train', 'test', 'validation'],
                        defaults=[tuple(), tuple(), tuple()])