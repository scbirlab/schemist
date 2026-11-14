"""Chemical structure cleaning routines."""

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from carabiner.decorators import return_none_on_error, vectorize

if TYPE_CHECKING:
    from rdkit.Chem import Mol
else:
    Mol = Any

DEFAULT_PH: float = 7.4


# @return_none_on_error
def sanitize_smiles_to_mol(s: str) -> Mol:
    r"""Parse SMILES without strict sanitization, apply sanifix5, then perceive stereochemistry.

    Parsing with sanitize=False skips the stereochemistry perception that
    MolFromSmiles normally performs, so double-bond (E/Z) stereo was left
    unassigned and chirality-aware fingerprints silently differed from RDKit's.
    Stereochemistry is now assigned after sanitization, as MolFromSmiles does.

    Examples
    ========
    >>> from rdkit.Chem import BondStereo, MolFromSmiles
    >>> m = sanitize_smiles_to_mol("C/C=C/C")
    >>> m.GetBondWithIdx(1).GetStereo() == BondStereo.STEREOE
    True
    >>> m = sanitize_smiles_to_mol("C/C=C\\C")
    >>> m.GetBondWithIdx(1).GetStereo() == BondStereo.STEREOZ
    True
    >>> m = sanitize_smiles_to_mol("C[C@H](N)C(=O)O")
    >>> m.GetAtomWithIdx(1).GetProp("_CIPCode")
    'S'
    >>> from rdkit.Chem import rdFingerprintGenerator
    >>> g = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
    >>> s = "CC(=O)N/C=C/c1ccccc1"
    >>> g.GetFingerprint(sanitize_smiles_to_mol(s)) == g.GetFingerprint(MolFromSmiles(s))
    True
    >>> from rdkit.Chem import MolToInchiKey
    >>> MolToInchiKey(sanitize_smiles_to_mol("C/C=C/C"))
    'IAQRGUVFOMOMEM-ONEGZZNKSA-N'
    >>> MolToInchiKey(sanitize_smiles_to_mol("C/C=C\\C"))
    'IAQRGUVFOMOMEM-ARJAWSKDSA-N'

    """
    from rdkit.Chem import AssignStereochemistry, MolFromSmiles
    from .sanifix5 import sanifix
    m = sanifix(MolFromSmiles(s, sanitize=False))
    if m is not None:
        AssignStereochemistry(m, cleanIt=True, force=True)
    return m


def _none_on_error(f):
    """Like ``return_none_on_error``, but missing optional dependencies still raise."""
    from functools import wraps

    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ImportError:
            raise
        except Exception:
            return None

    return wrapped


@dataclass(frozen=True)
class MolGroup:
    """A parent fragment and the other fragments removed from it.

    Holds RDKit ``Mol`` objects when returned by :func:`clean_mol`, and canonical
    SMILES when returned by :func:`clean_smiles`.

    Attributes
    ==========
    parent : Mol or str
        The parent fragment.
    other : tuple
        Every other fragment (counterions, solvents), in input order.

    """
    parent: Any
    other: tuple = ()

    @property
    def other_smiles(self) -> str | None:
        """Other fragments as one dot-separated SMILES, or None if there are none."""
        if not self.other:
            return None
        from rdkit.Chem import MolToSmiles
        return ".".join(x if isinstance(x, str) else MolToSmiles(x) for x in self.other)


def _protonate_smiles(
    smiles: str, 
    pH: float = DEFAULT_PH
) -> str:
    """Return the single dominant protonation state at ``pH`` with Dimorphite-DL.

    With ``ph_min == ph_max`` and ``precision=0``, each ionisable site is
    assigned deterministically by comparing its predicted pKa with ``pH``.

    """
    try:
        from dimorphite_dl import protonate_smiles
    except ImportError as e:
        raise ImportError(
            "Protonation requires Dimorphite-DL: pip install schemist[protonate]"
        ) from e
    from loguru import logger
    logger.disable("dimorphite_dl")
    states = protonate_smiles(
        smiles,
        ph_min=pH,
        ph_max=pH,
        precision=0.0,
        max_variants=1,
    )
    if not states:
        raise ValueError(f"Dimorphite-DL returned no protonation state for {smiles}")
    return states[0]


@_none_on_error
def clean_mol(
    mol: Mol,
    skip_desalt: bool = False,
    keep_H: bool = False,
    protonate: bool = False,
    protonation_pH: float = DEFAULT_PH
) -> MolGroup | None:
    """Desalt, remove explicit hydrogens and optionally protonate a sanitized molecule.

    Any input representation can be cleaned this way once parsed to a ``Mol``.
    Steps, in order:

    1. Unless ``skip_desalt``, keep the parent fragment with RDKit's
       ``rdMolStandardize.FragmentParent`` (strips known salts and solvents,
       then keeps the largest remaining fragment). Charges are not neutralised.
    2. Unless ``keep_H``, remove explicit hydrogen atoms, as ``MolFromSmiles`` does.
    3. If ``protonate``, set the dominant protonation state at ``protonation_pH``
       with Dimorphite-DL (optional dependency: ``schemist[protonate]``).
       Protonation implies ``keep_H=True``.

    Parameters
    ==========
    mol : Mol
        A sanitized RDKit molecule, or None.
    skip_desalt : bool, optional
        Keep all fragments in the parent. Default: ``False`` (desalt).
    keep_H : bool, optional
        Keep explicit hydrogen atoms. Default: ``False``.
    protonate : bool, optional
        Set the protonation state at ``protonation_pH``. Default: ``False``.
    protonation_pH : float, optional
        pH for ``protonate``. Default: 7.4.

    Returns
    =======
    MolGroup or None
        Parent ``Mol`` and the removed fragment ``Mol`` objects; ``None`` if
        ``mol`` is None or cleaning fails.

    Examples
    ========
    >>> group = clean_mol(sanitize_smiles_to_mol("CN.Cl"))
    >>> from rdkit.Chem import MolToSmiles
    >>> MolToSmiles(group.parent), group.other_smiles
    ('CN', 'Cl')
    >>> from rdkit.Chem import rdMolDescriptors
    >>> group = clean_mol(sanitize_smiles_to_mol("NCc1ccccc1.Cl"), keep_H=True)
    >>> round(rdMolDescriptors.CalcTPSA(group.parent), 2)
    26.02
    >>> clean_mol(None) is None
    True

    """
    from rdkit.Chem import GetMolFrags, GetSymmSSSR, MolToSmiles, RemoveHs
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if mol is None:
        return None
    if protonate:
        keep_H = True

    if skip_desalt:
        parent, others = mol, []
    else:
        parent = rdMolStandardize.FragmentParent(mol, skipStandardize=True)
        # FragmentParent returns a molecule without ring information, which
        # ring-dependent outputs (scaffolds, TPSA, fingerprints) need.
        parent.UpdatePropertyCache(strict=False)
        GetSymmSSSR(parent)
        parent_smiles = MolToSmiles(parent)
        others = [
            frag for frag in GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if MolToSmiles(frag) != parent_smiles
        ]

    if not keep_H:
        parent = RemoveHs(parent)
        others = [RemoveHs(frag) for frag in others]
    if protonate:
        parent = sanitize_smiles_to_mol(
            _protonate_smiles(MolToSmiles(parent), pH=protonation_pH)
        )
        if parent is None:
            raise ValueError("Protonated SMILES could not be parsed.")
    return MolGroup(parent=parent, other=tuple(others))


@vectorize
@_none_on_error
def clean_smiles(
    smiles: str,
    *args,
    skip_desalt: bool = False,
    return_all_fragments: bool = False,
    keep_H: bool = False,
    protonate: bool = False,
    protonation_pH: float = DEFAULT_PH,
    **kwargs,
) -> str | MolGroup:
    """Sanitize, desalt and optionally protonate a SMILES string or list of SMILES strings.

    Parses with :func:`sanitize_smiles_to_mol` (sanifix + stereo perception),
    then cleans with :func:`clean_mol`.

    Parameters
    ==========
    smiles : str or Iterable[str]
        Input SMILES.
    *args, **kwargs
        Passed to ``rdkit.Chem.MolToSmiles``.
    skip_desalt : bool, optional
        Keep all fragments in the output. Default: ``False`` (desalt).
    return_all_fragments : bool, optional
        Return a :class:`MolGroup` of canonical SMILES with the parent and the
        removed fragments. Default: ``False``.
    keep_H : bool, optional
        Keep explicit hydrogen atoms from the input. Default: ``False``.
    protonate : bool, optional
        Set the protonation state at ``protonation_pH``. Implies ``keep_H``.
        Default: ``False``.
    protonation_pH : float, optional
        pH for ``protonate``. Default: 7.4.

    Returns
    =======
    str or MolGroup or None
        Canonical SMILES (or a MolGroup); ``None`` if the input cannot be parsed.

    Examples
    ========
    >>> clean_smiles("CN.Cl")
    'CN'
    >>> clean_smiles("CN.Cl", skip_desalt=True)
    'CN.Cl'
    >>> clean_smiles("CC(=O)[O-].[Na+]", return_all_fragments=True)
    MolGroup(parent='CC(=O)[O-]', other=('[Na+]',))
    >>> clean_smiles("CN.Cl", return_all_fragments=True).other_smiles
    'Cl'
    >>> clean_smiles("CCO", return_all_fragments=True)
    MolGroup(parent='CCO', other=())
    >>> clean_smiles("[H]OC([H])([H])C")
    'CCO'
    >>> clean_smiles("[H]OC([H])([H])C", keep_H=True)
    '[H]OC([H])([H])C'
    >>> list(clean_smiles(["CN.Cl", "not_a_smiles"]))
    ['CN', None]

    """
    from rdkit.Chem import MolToSmiles

    group = clean_mol(
        sanitize_smiles_to_mol(smiles),
        skip_desalt=skip_desalt,
        keep_H=keep_H,
        protonate=protonate,
        protonation_pH=protonation_pH,
    )
    if group is None:
        return None
    parent = MolToSmiles(group.parent, *args, **kwargs)
    if return_all_fragments:
        return MolGroup(
            parent=parent,
            other=tuple(MolToSmiles(f, *args, **kwargs) for f in group.other),
        )
    return parent


@vectorize
def clean_selfies(
    selfies: str, 
    *args, 
    **kwargs
) -> str:
    """Sanitize a SELFIES string or list of SELFIES strings.
    
    """
    from rdkit.Chem import MolToSmiles
    import selfies as sf
    return sf.encode(MolToSmiles(sanitize_smiles_to_mol(sf.decode(selfies), *args, **kwargs)))
