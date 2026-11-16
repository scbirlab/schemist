"""3D conformer generation."""

from typing import TYPE_CHECKING, Any
import math

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from rdkit.Chem import Mol
else:
    Mol = Any

ETKDG_VERSIONS: tuple[int, ...] = (2, 3, 4)


def validate_conformer_args(
    n_conformers: int,
    seed: int,
    etkdg_version: int,
    prune_rms_thresh: float,
    minimize: bool,
    max_iters: int,
    n_threads: int
) -> None:
    """Check conformer-generation arguments, as used by :func:`embed_mol_conformers`.

    Public so that callers can validate a configuration once, before
    embedding many molecules, and so tell configuration errors apart from
    per-molecule failures.

    Raises
    ------
    ValueError
        If any argument is invalid.

    Examples
    --------
    >>> validate_conformer_args(1, 42, 3, -1., True, 500, 1)
    >>> validate_conformer_args(1, -1, 3, -1., True, 500, 1)
    Traceback (most recent call last):
        ...
    ValueError: seed must be an int in [0, 2**31), got -1.

    """
    def _int(x) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if not _int(n_conformers) or n_conformers < 1:
        raise ValueError(f"n_conformers must be an int >= 1, got {n_conformers!r}.")
    if not _int(seed) or not 0 <= seed < 2 ** 31:
        raise ValueError(f"seed must be an int in [0, 2**31), got {seed!r}.")
    if not _int(etkdg_version) or etkdg_version not in ETKDG_VERSIONS:
        raise ValueError(f"etkdg_version must be one of {ETKDG_VERSIONS}, got {etkdg_version!r}.")
    if (
        not isinstance(prune_rms_thresh, (int, float))
        or isinstance(prune_rms_thresh, bool)
        or not math.isfinite(prune_rms_thresh)
    ):
        raise ValueError(
            f"prune_rms_thresh must be a finite number (negative disables pruning), "
            f"got {prune_rms_thresh!r}."
        )
    if minimize and (not _int(max_iters) or max_iters < 1):
        raise ValueError(f"max_iters must be an int >= 1 when minimizing, got {max_iters!r}.")
    if not _int(n_threads) or n_threads < 0:
        raise ValueError(f"n_threads must be an int >= 0, got {n_threads!r}.")
    return None


def embed_mol_conformers(
    mol: "Mol",
    n_conformers: int = 1,
    seed: int = 42,
    etkdg_version: int = 3,
    prune_rms_thresh: float = -1.,
    minimize: bool = True,
    max_iters: int = 500,
    n_threads: int = 1
) -> tuple["Mol", list[int]]:
    """Add hydrogens and embed 3D conformers into an RDKit Mol.

    The core routine for conformer generation in schemist: conformers are
    embedded with ETKDG (`etkdg_version`, random seed `seed`) and, if
    `minimize`, relaxed with MMFF94 for up to `max_iters` steps; a conformer
    that has not converged keeps its last coordinates.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule, without explicit hydrogens. Not modified.
    n_conformers : int
        Conformers to embed. Default: 1.
    seed : int
        ETKDG random seed. Default: 42.
    etkdg_version : int
        ETKDG version: 2, 3 or 4. Default: 3.
    prune_rms_thresh : float
        RMSD (Å) below which conformers are pruned as duplicates; negative
        disables pruning. Default: -1 (no pruning).
    minimize : bool
        Relax conformers with MMFF94. Default: True.
    max_iters : int
        Maximum MMFF94 steps. Default: 500.
    n_threads : int
        Threads for embedding and minimization; 0 uses all cores. Default: 1.

    Conformers that exhaust `max_iters` keep their last coordinates; MMFF94
    convergence is not reported.

    Returns
    -------
    tuple
        The molecule with hydrogens and conformers, and the conformer IDs
        (at least one; fewer than `n_conformers` if some were pruned or
        failed to embed).

    Raises
    ------
    ValueError
        If arguments are invalid: `n_conformers` < 1, `seed` outside
        [0, 2**31) (RDKit's -1, a random seed, is not reproducible),
        `etkdg_version` not 2, 3 or 4, `max_iters` < 1 with `minimize`,
        or `n_threads` < 0.
    ImportError
        If the installed RDKit lacks the requested ETKDG version.
    RuntimeError
        If no conformer can be embedded, or (with `minimize`) MMFF94 has no
        parameters for some atoms, so that conformers are never left
        unminimized.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mol, ids = embed_mol_conformers(Chem.MolFromSmiles("CCO"), n_conformers=3)
    >>> mol.GetNumAtoms(), ids
    (9, [0, 1, 2])
    >>> embed_mol_conformers(Chem.MolFromSmiles("C[Si](C)(C)O[B](O)O"))
    Traceback (most recent call last):
        ...
    RuntimeError: MMFF94 has no parameters for some atoms
    >>> len(embed_mol_conformers(Chem.MolFromSmiles("C[Si](C)(C)O[B](O)O"), minimize=False)[1])
    1

    RMS pruning can return fewer conformers than requested:

    >>> len(embed_mol_conformers(Chem.MolFromSmiles("c1ccccc1"), n_conformers=5, prune_rms_thresh=0.5)[1]) < 5
    True

    Invalid arguments raise `ValueError`, before RDKit sees them:

    >>> for bad in ({"seed": -1}, {"seed": 2 ** 31}, {"n_threads": -1}, {"max_iters": 0},
    ...             {"etkdg_version": 3.0}, {"prune_rms_thresh": float("nan")}):
    ...     try:
    ...         embed_mol_conformers(Chem.MolFromSmiles("CCO"), **bad)
    ...     except ValueError as e:
    ...         print(e)
    seed must be an int in [0, 2**31), got -1.
    seed must be an int in [0, 2**31), got 2147483648.
    n_threads must be an int >= 0, got -1.
    max_iters must be an int >= 1 when minimizing, got 0.
    etkdg_version must be one of (2, 3, 4), got 3.0.
    prune_rms_thresh must be a finite number (negative disables pruning), got nan.

    An RDKit without the requested ETKDG version gives an informative error:

    >>> from unittest import mock
    >>> from rdkit.Chem import AllChem
    >>> with mock.patch.object(AllChem, "ETKDGv4", None):
    ...     embed_mol_conformers(Chem.MolFromSmiles("CCO"), etkdg_version=4)
    Traceback (most recent call last):
        ...
    ImportError: RDKit ... has no ETKDGv4, needed for etkdg_version=4. Use an older ETKDG version, or install a newer RDKit (schemist declares rdkit>=2026.09.1).

    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    validate_conformer_args(n_conformers, seed, etkdg_version, prune_rms_thresh, minimize, max_iters, n_threads)
    factory = getattr(AllChem, f"ETKDGv{etkdg_version}", None)
    if factory is None:  # e.g. the old_glibc extra's RDKit has no ETKDGv4
        import rdkit
        raise ImportError(
            f"RDKit {rdkit.__version__} has no ETKDGv{etkdg_version}, needed for "
            f"etkdg_version={etkdg_version}. Use an older ETKDG version, or "
            "install a newer RDKit (schemist declares rdkit>=2026.09.1)."
        )

    mol = Chem.AddHs(mol)
    params = factory()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms_thresh
    params.numThreads = n_threads
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if len(conformer_ids) == 0:
        raise RuntimeError(f"No conformers embedded (of {n_conformers})")
    if minimize:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise RuntimeError("MMFF94 has no parameters for some atoms")
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=n_threads, maxIters=max_iters)
    return mol, conformer_ids


def embed_conformers(
    smiles: str,
    n_conformers: int = 1,
    seed: int = 42,
    etkdg_version: int = 3,
    prune_rms_thresh: float = -1.,
    minimize: bool = True,
    max_iters: int = 500,
    n_threads: int = 1
) -> tuple[NDArray, NDArray]:
    """Embed 3D conformers, with hydrogens, for one SMILES string, as arrays.

    The SMILES is parsed with plain RDKit `MolFromSmiles`, without cleaning,
    because conformers depend on atom order; conformers are generated with
    :func:`embed_mol_conformers` (see there for the options and defaults).

    Returns
    -------
    tuple of NDArray
        Atomic numbers, int8 [N], and coordinates in Å, float32 [C, N, 3].
        C is at least 1, and fewer than `n_conformers` if some were pruned
        or failed to embed.

    Raises
    ------
    ValueError
        If the SMILES is missing or cannot be parsed, or arguments are invalid.
    RuntimeError
        If no conformer can be embedded, or (with `minimize`) MMFF94 has no
        parameters for some atoms.

    Examples
    --------
    >>> z, xyz = embed_conformers("CCO", n_conformers=3)
    >>> z.tolist()
    [6, 6, 8, 1, 1, 1, 1, 1, 1]
    >>> xyz.shape, xyz.dtype
    ((3, 9, 3), dtype('float32'))
    >>> bool((embed_conformers("CCO", n_conformers=3)[1] == xyz).all())  # deterministic
    True
    >>> embed_conformers("not-a-smiles")
    Traceback (most recent call last):
        ...
    ValueError: Invalid SMILES: 'not-a-smiles'
    >>> embed_conformers("[Fe]")
    Traceback (most recent call last):
        ...
    RuntimeError: '[Fe]': MMFF94 has no parameters for some atoms
    >>> embed_conformers("[Fe]", minimize=False)[1].shape
    (1, 1, 3)
    >>> bool((embed_conformers("CCO", n_conformers=3, seed=7)[1] != xyz).any())  # seed matters
    True

    RDKit's global logging settings are left as they were:

    >>> from rdkit import rdBase, RDLogger
    >>> before = dict(line.split(":") for line in rdBase.LogStatus().splitlines())
    >>> RDLogger.DisableLog("rdApp.error")
    >>> status = rdBase.LogStatus()
    >>> _ = embed_conformers("CCO")
    >>> embed_conformers("not-a-smiles")
    Traceback (most recent call last):
        ...
    ValueError: Invalid SMILES: 'not-a-smiles'
    >>> rdBase.LogStatus() == status
    True
    >>> for level, state in before.items():  # restore the original settings
    ...     (RDLogger.EnableLog if state == "enabled" else RDLogger.DisableLog)(level)

    """
    from rdkit import Chem

    if not isinstance(smiles, str) or len(smiles) == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)  # RDKit logging is left to the caller
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    try:
        mol, conformer_ids = embed_mol_conformers(
            mol,
            n_conformers=n_conformers,
            seed=seed,
            etkdg_version=etkdg_version,
            prune_rms_thresh=prune_rms_thresh,
            minimize=minimize,
            max_iters=max_iters,
            n_threads=n_threads,
        )
    except RuntimeError as e:
        raise RuntimeError(f"{smiles!r}: {e}") from e

    atomic_numbers = np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int8)
    coordinates = np.stack([
        mol.GetConformer(cid).GetPositions() for cid in conformer_ids
    ]).astype(np.float32)
    return atomic_numbers, coordinates
