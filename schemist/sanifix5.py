"""
Updates to sanifix4.py (allowing wierd valencies)
Original code from rdkit [James Davidson]
"""

from carabiner import print_err
from rdkit import RDLogger
from rdkit.Chem import (
    EditableMol,
    GetMolFrags,
    GetSymmSSSR,
    KekulizeException,
    Mol,
    MolFromSmarts,
    MolToSmiles,
    SanitizeFlags,
    SanitizeMol,
)

RDLogger.DisableLog('rdApp.*')

def _FragIndicesToMol(oMol, indices):
    em = EditableMol(Mol())

    newIndices = {}
    for i, idx in enumerate(indices):
        em.AddAtom(oMol.GetAtomWithIdx(idx))
        newIndices[idx] = i

    for i, idx in enumerate(indices):
        at = oMol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx() == idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx < idx:
                continue
            em.AddBond(newIndices[idx], newIndices[oidx], bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap = newIndices
    return res


def _recursivelyModifyNs(mol: Mol, matches, indices=None):
    if indices is None:
        indices = []
    res = None
    while len(matches) and res is None:
        tIndices = indices[:]
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Mol(mol.ToBinary())
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
        cp = Mol(nm.ToBinary())
        try:
            SanitizeMol(cp)
        except ValueError:
            res, indices = _recursivelyModifyNs(nm, matches, indices=tIndices)
        else:
            indices = tIndices
            res = cp
    return res, indices


def AdjustAromaticNs(m: Mol, nitrogenPattern:str = "[n&D2&H0;r5,r6]") -> Mol:
    """
    default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
    to fix: O=c1ccncc1
    """
    GetSymmSSSR(m)
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings:
    em = EditableMol(m)
    linkers = m.GetSubstructMatches(MolFromSmarts("[r]!@[r]"))
    plsFix = set()
    for a, b in linkers:
        em.RemoveBond(a, b)
        plsFix.add(a)
        plsFix.add(b)
    nm = em.GetMol()
    for at in plsFix:
        at = nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum() == 7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = GetMolFrags(nm)
    frags = [_FragIndicesToMol(nm, x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    ok = True
    for i, frag in enumerate(frags):
        cp = Mol(frag)
        try:
            SanitizeMol(cp)
        except ValueError:
            matches = [x[0] for x in frag.GetSubstructMatches(MolFromSmarts(nitrogenPattern))]
            lres, indices = _recursivelyModifyNs(frag, matches)
            if not lres:
                # print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                ok = False
                break
            else:
                revMap = {}
                for k, v in frag._idxMap.items():
                    revMap[v] = k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)
    if not ok:
        return None
    return m


def sanitize_allowing_valence_errors(m: Mol) -> Mol:
    """Allow valence errors but check everything else."""
    m.UpdatePropertyCache(strict=False)
    SanitizeMol(
        m,
        SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES,
    )
    return m


def sanifix(m: Mol) -> Mol:
    if m is None:
        return None
    try:
        m.UpdatePropertyCache(False)
        cp = Mol(m.ToBinary())
        sanitize_allowing_valence_errors(cp)
    except ValueError as e:
        print_err(f"{MolToSmiles(m)}: attempting fix: {e}")
        try:
            m = AdjustAromaticNs(m)
            if m is not None:
                sanitize_allowing_valence_errors(m)
            return m
        except Exception as ee:
            print_err(f"{MolToSmiles(m)}: failed: {ee}")
            return None
    except RuntimeError as e:
        print_err(f"{MolToSmiles(m)}: failed: {e}")
        raise e
    else:
        return cp