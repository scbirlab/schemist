"""Types used in schemist."""

from typing import TYPE_CHECKING, Any, TypeAlias
from collections.abc import Iterable
from collections import namedtuple

if TYPE_CHECKING:
    from rdkit.Chem import Mol
else:
    Mol = Any

DataSplits = namedtuple(
    'DataSplits',
    ['train', 'test', 'validation'],
    defaults=[(), (), ()],
)

StrOrIterable: TypeAlias = str | Iterable[str] 
OptionalStrOrIterable: TypeAlias = str | None | Iterable[str | None]

OptionalMol: TypeAlias = Mol | None
MolOrIterable: TypeAlias = Mol | Iterable[Mol] 
OptionalMolOrIterable: TypeAlias = OptionalMol | Iterable[OptionalMol] 
