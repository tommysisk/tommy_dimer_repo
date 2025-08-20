from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdMolTransforms
import tempfile, os
from collections import defaultdict
from functools import partial
from writhe_tools.md_tools import canonical_residues, parallel_displacements, parallel_distances
from writhe_tools.writhe_ray import divnorm
from writhe_tools.utils import num_str, group_by


wrap_angle = lambda theta: np.minimum(theta, np.pi - theta)

rm_index = lambda x : np.array(list(map(partial(num_str, reverse=True), x)))

isin_index = lambda x, y : np.where(np.isin(x, y))[0]


# ── canonical aromatic rings (6 atoms each) ───────────────────────────────
CANONICAL: Dict[str, List[str]] = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CD1", "CG", "CZ2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"]  # imidazole padded
}

# ── internal helper: RDKit ring detection for one residue slice ───────────
def _rings_from_rdkit_slice(res_traj: md.Trajectory, ring_sizes=(5, 6)) -> List[List[int]]:
    """Return lists of atom indices (wrt slice) that form rings."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        fname = tmp.name
    try:
        res_traj[0].save_pdb(fname, force_overwrite=True)
        mol = Chem.MolFromPDBFile(fname, removeHs=False, sanitize=True)
        if mol is None:
            return []
        rings = []
        for ring in Chem.GetSymmSSSR(mol):
            if len(ring) in ring_sizes:
                rings.append([int(i) for i in ring])
        return rings
    finally:
        os.remove(fname)

# ── main vectorised geometry routine ──────────────────────────────────────
def aromatic_rings(
        traj: md.Trajectory,
        canonical: Dict[str, List[str]] = CANONICAL
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str, List[int]]]]:
    """
    Full pipeline: canonical + RDKit-detected rings → centroids & normals.

    Returns
    -------
    centroids : (n_frames, n_rings, 3)
    normals   : (n_frames, n_rings, 3)
    ring_ids  : list[(res_index, res_name, atom_idx_list)]
    """
    top = traj.topology
    ring_indices: List[List[int]] = []
    ring_ids:     List[Tuple[int, str, List[int]]] = []
    centroids = []
    for res in top.residues:  # zero-indexed by default: res.index
        # --- 1) canonical amino-acid rings --------------------------------
        if res.name in canonical:
            try:
                idx = [res.atom(n).index for n in canonical[res.name]]
                ring_indices.append(idx[:3])
                ring_ids.append((res.index, res.name, idx))
                centroids.append(traj.xyz[:, np.array(idx), :].mean(1))
            except KeyError:
                pass  # skip if atoms missing
            continue  # canonical handled

        # --- 2) non-canonical: RDKit ring detection -----------------------
        res_atoms = [a.index for a in res.atoms]
        if not res_atoms:
            continue
        slice_traj = traj.atom_slice(res_atoms, inplace=False)
        for ring in _rings_from_rdkit_slice(slice_traj):
            full_idx = [res_atoms[i] for i in ring]
            centroids.append(traj.xyz[:, np.array(full_idx), :].mean(1))
            ring_indices.append(full_idx[:3])
            ring_ids.append((res.index, res.name, full_idx))

    if not ring_indices:
        raise ValueError("No aromatic or ligand rings detected.")

    ring_indices = np.asarray(ring_indices)               # (r,6)
    xyz  = traj.xyz[:, ring_indices, :]                   # (f,r,6,3)
    box  = traj.unitcell_lengths[:, None, :]              # (f,1,3)

    # centroids
    centroids = np.stack(centroids, 1)                         # (f,r,3)

    # minimum-image displacement vectors
    v1 = xyz[:, :, 1] - xyz[:, :, 0]
    v2 = xyz[:, :, 2] - xyz[:, :, 0]
    v1 -= box * np.round(v1 / box)
    v2 -= box * np.round(v2 / box)

    return centroids, divnorm(np.cross(v1, v2)), ring_ids



def aro_angles(pairs, centers, normals,  unitcell_lengths):
    displacements = parallel_displacements(centers, pairs, unitcell_lengths) # displacements corrected for PBC
    distances = np.linalg.norm(displacements, axis=-1, keepdims=True) # constract displacements to distances
    displacements /= distances # norm displacements

    #compute angles using broadcasted dot products
    theta = np.arccos(np.sum(normals[:, pairs.T[0]] * normals[:, pairs.T[1]], axis=-1))
    phi = np.arccos(np.sum(displacements[:, pairs.T[0]] * normals[:, pairs.T[1]], axis=-1))

    return *[wrap_angle(i) for i in (theta, phi)], distances.squeeze()



import numpy as np
from typing import Iterable, Dict, List, Tuple, Optional

class Aromatics:
    """
    Broadcasted aromatic angle/distance computation using your helpers.

    __init__(traj):
        calls `aromatic_rings(traj)` → centers(F,R,3), normals(F,R,3), ids
        builds ring/residue maps (ring → res, ordinal within residue, names, etc.)

    angles_between_groups(groupA, groupB) -> (angles, group_ids, pair_names)
        groupA/groupB: iterable of strings ("TYR", "45", "TYR:45", ...)

        angles     : (K, F, 3) with [theta(norm_i·norm_j), phi(u_ij·norm_j), dist_ij]
        group_ids  : (K,) ints, identical for rows of the same residue–residue pair
        pair_names : (K, 2) strings: "RES:idx" or "RES:idx:ring"
                     (ring suffix only when residue has multiple rings)
    """

    def __init__(self, traj):
        self.box  = traj.unitcell_lengths  # (F, 3)
        self.n_frames = traj.n_frames

        # --- Use your ring finder directly ---
        self.centers, self.normals, self.ids = aromatic_rings(traj)
        # ids: list of (res_index, res_name, atom_idx_list)

        # Build ring -> residue index and ring's ordinal within residue
        self.rings_per_res: Dict[int, List[int]] = {}
        self.res_name: Dict[int, str] = {}
        for ring_idx, (res_idx, res_nm, _full) in enumerate(self.ids):
            self.rings_per_res.setdefault(res_idx, []).append(ring_idx)
            self.res_name[res_idx] = res_nm

        # Within-residue ring ordinals
        self.ring_to_res: Dict[int, int] = {}
        self.ring_ord_in_res: Dict[int, int] = {}
        for res_idx, ring_list in self.rings_per_res.items():
            for ord_in_res, ring_idx in enumerate(ring_list):
                self.ring_to_res[ring_idx] = res_idx
                self.ring_ord_in_res[ring_idx] = ord_in_res

        # Residue indices by NAME (for selection)
        self.res_indices_by_name: Dict[str, List[int]] = {}
        for ri, rn in self.res_name.items():
            self.res_indices_by_name.setdefault(rn.upper(), []).append(ri)

    # ------------------------ PUBLIC API ------------------------

    def angles_between_groups(
            self,
            groupA: Iterable[str],
            groupB: Iterable[str],
            wrap: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fully broadcasted computation (single PBC displacement call).
        Returns (angles, group_ids, pair_names) with shapes described above.
        """
        # Resolve residue selections
        resA = self._resolve_to_res_indices(groupA)
        resB = self._resolve_to_res_indices(groupB)

        # Rings for each residue set
        ringsA = [r for ra in resA for r in self.rings_per_res.get(ra, [])]
        ringsB = [r for rb in resB for r in self.rings_per_res.get(rb, [])]

        Na, Nb = len(ringsA), len(ringsB)
        if Na == 0 or Nb == 0:
            return (np.empty((0, self.n_frames, 3), dtype=float),
                    np.empty((0,), dtype=int),
                    np.empty((0, 2), dtype=object))

        ringsA = np.asarray(ringsA, dtype=int)
        ringsB = np.asarray(ringsB, dtype=int)

        # ---------- Core math (broadcasted, memory-aware) ----------
        # Build pair index table once (K = Na*Nb)
        pairs = np.column_stack([np.repeat(ringsA, Nb), np.tile(ringsB, Na)])  # (K, 2)
        K = pairs.shape[0]

        # 1) PBC displacements for centroids (i -> j), then distance and in-place normalize
        U = parallel_displacements(self.centers, pairs, self.box)  # (F, K, 3)
        dist = np.linalg.norm(U, axis=-1)                          # (F, K)
        with np.errstate(invalid='ignore', divide='ignore'):
            U /= np.where(dist[..., None] == 0, 1.0, dist[..., None])

        # Reshape to broadcast with normals picked by ringsA / ringsB
        # U_reshaped: (F, Na, Nb, 3)
        U = U.reshape(self.n_frames, Na, Nb, 3)

        # Normals for A and B selections (small views, no copies of the big pair tensor)
        nA = self.normals[:, ringsA, :]   # (F, Na, 3)
        nB = self.normals[:, ringsB, :]   # (F, Nb, 3)

        # 2) theta = arccos( nA · nB )  → shape (F, Na, Nb)
        # Minimal fix: expand dims so shapes are (F, Na, 1, 3) and (F, 1, Nb, 3)
        cos_theta = np.sum(nA[:, :, None, :] * nB[:, None, :, :], axis=-1).clip(-1, 1)  # (F, Na, Nb)
        theta = np.arccos(cos_theta)

        # 3) phi = arccos( u_ij · nB ), broadcasting nB across Na
        #    U: (F, Na, Nb, 3), nB[:, None, :, :] -> (F, 1, Nb, 3)
        phi = np.arccos(np.sum(U * nB[:, None, :, :], axis=-1).clip(-1, 1))  # (F, Na, Nb)

        if wrap:
            theta, phi = (wrap_angle(i) for i in (theta, phi))

        # Flatten (F, Na, Nb) -> (F, K) and stack to (K, F, 3)
        angles = np.stack([i.reshape(self.n_frames, K) for i in [theta, phi, dist]], axis=-1).transpose((1, 0, 2))
        # (K, F, 3)

        # ---------- Lean labeling & grouping (no big temp arrays) ----------
        # Precompute ring labels (site-aware) and base residue labels
        ring_label_site  = {}
        ring_label_base  = {}
        # Count rings per residue to decide whether to include :ring suffix
        rings_per_res_counts = {ri: len(rlist) for ri, rlist in self.rings_per_res.items()}

        for r in np.unique(np.concatenate([ringsA, ringsB])):
            ri = self.ring_to_res[r]
            base = f"{self.res_name[ri]}:{ri}"
            if rings_per_res_counts[ri] > 1:
                lbl = f"{base}:{self.ring_ord_in_res[r]}"
            else:
                lbl = base
            ring_label_site[r] = lbl
            ring_label_base[r] = base

        # Build pair_names and group_ids with small loops
        pair_names: List[Tuple[str, str]] = []
        group_ids:  List[int] = []
        id_map: Dict[Tuple[str, str], int] = {}

        # Iterate in the same order as pairs were constructed (repeat A, tile B)
        for ra in ringsA.tolist():
            base_a = ring_label_base[ra]
            site_a = ring_label_site[ra]
            for rb in ringsB.tolist():
                base_b = ring_label_base[rb]
                gid = id_map.setdefault((base_a, base_b), len(id_map))
                group_ids.append(gid)
                pair_names.append((site_a, ring_label_site[rb]))

        pair_names = np.array(pair_names, dtype=object)  # (K, 2)
        group_ids  = np.array(group_ids, dtype=int)      # (K,)

        return angles, group_ids, pair_names

    # ------------------------- helpers -------------------------

    def _resolve_to_res_indices(self, items: Iterable[str]) -> List[int]:
        """
        Expand to residue indices present in this trajectory.
        Accepts: 'NAME', 'INDEX', 'NAME:INDEX', 'NAME:INDEX:CHAIN'.
        """
        out = set()
        for s in items:
            s_str = str(s).strip()
            s_up  = s_str.upper()

            # NAME
            if s_up in self.res_indices_by_name:
                out.update(self.res_indices_by_name[s_up])
                continue

            # pure INDEX
            idx = self._try_int(s_str)
            if idx is not None and idx in self.rings_per_res:
                out.add(idx)
                continue

            # NAME:INDEX(:CHAIN) -> use INDEX
            parts = s_str.split(':')
            if len(parts) >= 2:
                idx2 = self._try_int(parts[1])
                if idx2 is not None and idx2 in self.rings_per_res:
                    out.add(idx2)

        return sorted(out)

    @staticmethod
    def _try_int(x) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None


# ------------------------ tiny broadcast sanity test ------------------------
# This only checks the fixed broadcasting logic for theta/phi shapes, not MD specifics.


# corrected cutoffs - previous misses T stacks with probability 1/2

def t_stack(theta, phi, distance):
    return (theta > (5 / 12) * np.pi) & ((phi > np.pi / 3) | (phi < np.pi / 6)) & (distance < 0.75)

def pi_stack(theta, phi, distance):
    return (theta < np.pi / 4) & (phi < np.pi / 3) & (distance < 0.65)



def extract_hbond_sites(
        traj: md.Trajectory,
        canonical: Set[str] = None,
        fdef_path: str = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
) -> Tuple[
    List[Tuple[int, str, List[int]]],              # acceptors: single atom indices
    List[Tuple[int, str, List[List[int]]]]         # donors: [heavy_idx, H_idx] per donor
]:
    """
    Extract H-bond acceptors and donors from trajectory for both canonical residues and ligands.
    For ligands, RDKit is used to reconstruct connectivity if needed and to find heavy→H donor pairs.
    """
    top = traj.topology
    fdef = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    if canonical is None:
        canonical = canonical_residues  # from your environment

    acceptor_dict = defaultdict(list)
    donor_dict = defaultdict(list)

    for res in top.residues:
        res_idx = res.index
        res_name = res.name
        atoms = list(res.atoms)
        atom_indices = [a.index for a in atoms]
        if not atom_indices:
            continue

        # ----------------------------- Canonical residues (minimal S tweak) -----------------------------
        if res_name in canonical:
            for atom in atoms:
                idx = atom.index
                sym = atom.element.symbol.upper()

                if sym in {"O", "N"}:
                    # acceptor
                    acceptor_dict[(res_idx, res_name)].append(idx)
                    # donors: any bonded hydrogens
                    for bond in top.bonds:
                        a, b = bond
                        if a.index == idx and b.element.symbol.upper() == "H":
                            donor_dict[(res_idx, res_name)].append([idx, b.index])
                        elif b.index == idx and a.element.symbol.upper() == "H":
                            donor_dict[(res_idx, res_name)].append([idx, a.index])

                elif sym == "S":
                    # NEW: thiol/thiolate handling with minimal diff
                    has_h = False
                    for bond in top.bonds:
                        a, b = bond
                        if a.index == idx and b.element.symbol.upper() == "H":
                            donor_dict[(res_idx, res_name)].append([idx, b.index])  # S–H donor
                            has_h = True
                        elif b.index == idx and a.element.symbol.upper() == "H":
                            donor_dict[(res_idx, res_name)].append([idx, a.index])  # S–H donor
                            has_h = True
                    if not has_h:
                        # Unprotonated sulfur (thiolate-like) → weak/edge-case acceptor
                        acceptor_dict[(res_idx, res_name)].append(idx)
            continue

        # ----------------------------- Ligands / noncanonical residues (unchanged) -----------------------------
        slice_traj = traj.atom_slice(atom_indices, inplace=False)

        # Write only the residue to a PDB file (keeps atom order for index mapping)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            fname = tmp.name
        try:
            slice_traj[0].save_pdb(fname, force_overwrite=True)

            # Load without sanitization; then reconstruct bonds if needed and sanitize
            mol = Chem.MolFromPDBFile(fname, removeHs=False, sanitize=False)
            if mol is None:
                continue

            # If bonds are missing/incorrect, have RDKit guess connectivity, then sanitize
            try:
                # Connect the dots guesses bonds based on distances
                Chem.rdmolops.ConnectTheDots(mol)
                Chem.SanitizeMol(mol)
            except Exception:
                # As a fallback, at least try to sanitize whatever we got
                Chem.SanitizeMol(mol, catchErrors=True)

            # RDKit features on the sanitized mol
            feats = fdef.GetFeaturesForMol(mol)

            # Quick accessors
            conf = mol.GetConformer()
            is_H = lambda i: (mol.GetAtomWithIdx(i).GetAtomicNum() == 1)

            for feat in feats:
                fam = feat.GetFamily()
                # map RDKit indices (within residue) -> traj indices
                ridxs = list(feat.GetAtomIds())
                tidxs = [atom_indices[i] for i in ridxs]

                if fam == 'Acceptor':
                    # Acceptors can surface as single atoms; add them all (de-dup later)
                    for ti in tidxs:
                        acceptor_dict[(res_idx, res_name)].append(ti)

                elif fam == 'Donor':
                    # RDKit usually gives HEAVY only; sometimes includes H as well.
                    heavy_rids = [ri for ri in ridxs if not is_H(ri)]
                    if not heavy_rids:
                        # (very rare) donor feature with only hydrogens -> skip
                        continue

                    for h_rid in heavy_rids:
                        # Prefer H neighbors from RDKit bonds
                        h_neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(h_rid).GetNeighbors()
                                       if nbr.GetAtomicNum() == 1]

                        # Fallback: distance-based H search (≤ 1.2 Å) if connectivity is still off
                        if not h_neighbors:
                            pos_heavy = conf.GetAtomPosition(h_rid)
                            for a in mol.GetAtoms():
                                if a.GetAtomicNum() != 1:
                                    continue
                                hid = a.GetIdx()
                                pos_h = conf.GetAtomPosition(hid)
                                # distance in Å
                                dx = pos_heavy.x - pos_h.x
                                dy = pos_heavy.y - pos_h.y
                                dz = pos_heavy.z - pos_h.z
                                if (dx*dx + dy*dy + dz*dz) <= (1.2 * 1.2):
                                    h_neighbors.append(hid)

                        # Map to trajectory indices and append donor pairs
                        heavy_tidx = atom_indices[h_rid]
                        for h_rid2 in h_neighbors:
                            donor_dict[(res_idx, res_name)].append([heavy_tidx, atom_indices[h_rid2]])

        finally:
            try:
                os.remove(fname)
            except OSError:
                pass

    # ---- Format as lists; de-duplicate acceptors per residue and sort ----
    acceptors = []
    for (ri, rn), a_idxs in acceptor_dict.items():
        # unique & sorted for stability
        ai_unique = sorted(set(a_idxs))
        acceptors.append((ri, rn, ai_unique))

    donors = [(ri, rn, d_pairs) for (ri, rn), d_pairs in donor_dict.items()]

    return acceptors, donors


def hbond_def(theta, distance):
    return (theta < np.pi / 6) & (distance < 0.35)



class HBonds:
    """
    Vectorized H-bond analyzer (fast core math + simple/lean label building).

    angles_between_groups(donor_group, acceptor_group) returns:
      angles     : (K, F, 3) with [theta(D→H, H→A), dist_HA, dist_DA]
      group_ids  : (K,)      ints; same id for rows with the *same residue–residue* pair
                              (site suffix ignored)
      pair_names : (K, 2)    strings with site-aware labels:
                              "RES:resIndex" or "RES:resIndex:siteIndex" (0-based per residue)
    """

    def __init__(self, traj):
        self.xyz  = traj.xyz                       # (F, N, 3)
        self.box  = traj.unitcell_lengths          # (F, 3)

        # ---- Use your extractor directly
        acceptors, donors = extract_hbond_sites(traj)
        # acceptors: [(res_idx, res_name, [acc_atom_idx, ...]), ...]
        # donors   : [(res_idx, res_name, [[D_idx, H_idx], ...]), ...]

        # Residue names present (union donors/acceptors)
        self.res_name: Dict[int, str] = {}
        for ri, rn, _ in acceptors: self.res_name[ri] = rn
        for ri, rn, _ in donors:    self.res_name[ri] = rn

        # Per-residue structures
        self.acceptors_by_res: Dict[int, List[int]] = {
            ri: list(acc_list) for (ri, _rn, acc_list) in acceptors
        }
        self.donors_by_res: Dict[int, List[Tuple[int, int]]] = {
            ri: [tuple(p) for p in pairs] for (ri, _rn, pairs) in donors
        }

        # Residue indices by NAME (for lookup)
        self.res_indices_by_name: Dict[str, List[int]] = {}
        for ri, rn in self.res_name.items():
            self.res_indices_by_name.setdefault(rn.upper(), []).append(ri)

        # ---- Flatten donors globally and precompute u_DH (F, Nd, 3)
        self.donors_array = np.array(
            [pair for _ri, _rn, pairs in donors for pair in pairs],
            dtype=int
        ) if donors else np.zeros((0, 2), dtype=int)

        # Map global donor index -> (res_idx, within-res donor ordinal)
        self.donor_g_to_res: Dict[int, int] = {}
        self.donor_g_ord_in_res: Dict[int, int] = {}
        g = 0
        for ri, _rn, pairs in donors:
            for ord_in_res, _ in enumerate(pairs):
                self.donor_g_to_res[g] = ri
                self.donor_g_ord_in_res[g] = ord_in_res
                g += 1

        if self.donors_array.shape[0] > 0:
            dh = parallel_displacements(self.xyz, self.donors_array, self.box)  # (F, Nd, 3) D->H
            self.u_DH = divnorm(dh)                                             # (F, Nd, 3)
        else:
            self.u_DH = np.zeros((len(self.xyz), 0, 3), dtype=float)

        # Acceptors: map atom -> (res_idx, ordinal) for labels
        self.acc_atom_to_res: Dict[int, int] = {}
        self.acc_atom_ord_in_res: Dict[int, int] = {}
        for ri, accs in self.acceptors_by_res.items():
            for ord_in_res, aidx in enumerate(accs):
                self.acc_atom_to_res[aidx] = ri
                self.acc_atom_ord_in_res[aidx] = ord_in_res

        # Site counts (to decide if we append :siteIndex)
        self.n_donor_sites_by_res = {ri: len(self.donors_by_res.get(ri, [])) for ri in self.res_name.keys()}
        self.n_acceptors_by_res   = {ri: len(self.acceptors_by_res.get(ri, [])) for ri in self.res_name.keys()}

    # ------------------------ PUBLIC API ------------------------

    def angles_between_groups(
            self,
            donor_group: Iterable[str],
            acceptor_group: Iterable[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast core math via broadcasting; simple, memory-lean label building.
        """
        donor_res    = self._resolve_to_res_indices(donor_group)
        acceptor_res = self._resolve_to_res_indices(acceptor_group)

        # Selected donors (global site indices) & acceptors (atom indices)
        donor_g_sel = np.fromiter(
            (g for ri in donor_res for g in self._donor_globals_for_res(ri)),
            dtype=int,
            count=sum(len(self.donors_by_res.get(ri, [])) for ri in donor_res)
        )
        acc_atoms_sel = np.fromiter(
            (a for rj in acceptor_res for a in self.acceptors_by_res.get(rj, [])),
            dtype=int,
            count=sum(len(self.acceptors_by_res.get(rj, [])) for rj in acceptor_res)
        )

        Nd, Na = donor_g_sel.size, acc_atoms_sel.size
        if Nd == 0 or Na == 0:
            return (np.empty((0, len(self.xyz), 3), dtype=float),
                    np.empty((0,), dtype=int),
                    np.empty((0, 2), dtype=object))

        # Build Cartesian index pairs ONCE for HA and DA
        D_idx = self.donors_array[donor_g_sel, 0]  # (Nd,)
        H_idx = self.donors_array[donor_g_sel, 1]  # (Nd,)
        pairs_HA = np.column_stack([np.repeat(H_idx, Na), np.tile(acc_atoms_sel, Nd)])  # (K,2)
        pairs_DA = np.column_stack([np.repeat(D_idx, Na), np.tile(acc_atoms_sel, Nd)])  # (K,2)
        K = pairs_HA.shape[0]

        # ------------ Core math (broadcasted, memory-aware) ------------
        # H->A displacement -> distance -> IN-PLACE normalize to u_HA
        HA = parallel_displacements(self.xyz, pairs_HA, self.box)     # (F, K, 3)
        dist_HA = np.linalg.norm(HA, axis=-1)                         # (F, K)
        with np.errstate(invalid='ignore', divide='ignore'):
            HA /= np.where(dist_HA[..., None] == 0, 1.0, dist_HA[..., None])

        # Reshape to (F, Nd, Na, 3) so we can broadcast with u_DH without repeating
        HA = HA.reshape(len(self.xyz), Nd, Na, 3)                # (F, Nd, Na, 3)
        u_DH = self.u_DH[:, donor_g_sel, :].reshape(len(self.xyz), Nd, 1, 3)

        # θ = ∠(D→H, H→A)
        cosang = np.sum(u_DH * HA, axis=-1)                           # (F, Nd, Na)
        np.clip(cosang, -1.0, 1.0, out=cosang)
        theta = np.arccos(cosang)                                     # (F, Nd, Na)

        # D->A distances (reshape after)
        DA = parallel_displacements(self.xyz, pairs_DA, self.box)     # (F, K, 3)
        dist_DA = np.linalg.norm(DA, axis=-1)                         # (F, K)

        # Flatten back to (F, K) to stack
        theta   = theta.reshape(len(self.xyz), K)
        dist_HA = dist_HA.reshape(len(self.xyz), K)

        # Stack to (K, F, 3) — reuse names to avoid more temps
        out = np.stack([theta, dist_HA, dist_DA], axis=-1)            # (F, K, 3)
        angles = np.transpose(out, (1, 0, 2))                         # (K, F, 3)

        # ------------ Lean label building (no huge repeats/tiles) ------------
        # Precompute per-donor and per-acceptor labels once (small lists)
        donor_base  = []
        donor_site  = []
        donor_res_idx = [self.donor_g_to_res[g] for g in donor_g_sel]
        for g_idx, ri in zip(donor_g_sel.tolist(), donor_res_idx):
            base = f"{self.res_name[ri]}:{ri}"
            if self.n_donor_sites_by_res[ri] > 1:
                base_site = f"{base}:{self.donor_g_ord_in_res[g_idx]}"
            else:
                base_site = base
            donor_base.append(base)     # no suffix (for grouping)
            donor_site.append(base_site)  # site-aware

        acc_base = []
        acc_site = []
        for a in acc_atoms_sel.tolist():
            rj = self.acc_atom_to_res[a]
            base = f"{self.res_name[rj]}:{rj}"
            if self.n_acceptors_by_res[rj] > 1:
                base_site = f"{base}:{self.acc_atom_ord_in_res[a]}"
            else:
                base_site = base
            acc_base.append(base)
            acc_site.append(base_site)

        # Build pair_names and group_ids with a tiny dict; no big arrays
        pair_names: List[Tuple[str, str]] = []
        group_ids:  List[int] = []
        id_map: Dict[Tuple[str, str], int] = {}

        for di, db in enumerate(donor_base):
            ds = donor_site[di]
            for aj, ab in enumerate(acc_base):
                key = (db, ab)  # base-only for grouping
                gid = id_map.setdefault(key, len(id_map))
                group_ids.append(gid)
                pair_names.append((ds, acc_site[aj]))

        pair_names = np.array(pair_names, dtype=object)      # (K, 2)
        group_ids  = np.array(group_ids, dtype=int)          # (K,)

        return angles, group_ids, pair_names

    # ------------------------- helpers -------------------------

    def hbonds_between_groups(self,
                              donor_group: Iterable[str]=None,
                              acceptor_group: Iterable[str]=None,
                              criteria: callable = hbond_def,
                              aggr: callable = np.any,
                              angles: tuple = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast core math via broadcasting; simple, memory-lean label building.
        """
        if angles is not None:
            hbonds = group_by(angles[1], criteria(*angles[0][...,:-1].T).T , partial(aggr, axis=0))
            return hbonds, *angles[1:]
        
        angles, pair_groups, pairs = self.angles_between_groups(donor_group, acceptor_group)
        hbonds = group_by(pair_groups, criteria(*angles[...,:-1].T).T , partial(aggr, axis=0))
        return hbonds, pair_groups, pairs




    def _donor_globals_for_res(self, res_idx: int) -> List[int]:
        """Return global donor-site indices for a residue (order consistent with donors_array)."""
        out = []
        g = 0
        for r, pairs in self.donors_by_res.items():
            if r == res_idx:
                out.extend(range(g, g + len(pairs)))
            g += len(pairs)
        return out

    def _resolve_to_res_indices(self, items: Iterable[str]) -> List[int]:
        """
        Expand to residue indices present in the trajectory.
        Accepts: 'NAME', 'INDEX', 'NAME:INDEX', 'NAME:INDEX:CHAIN'.
        """
        out = set()
        for s in items:
            s_str = str(s).strip()
            s_up  = s_str.upper()

            # NAME
            if s_up in self.res_indices_by_name:
                out.update(self.res_indices_by_name[s_up])
                continue

            # INDEX
            idx = self._try_int(s_str)
            if idx is not None and (idx in self.acceptors_by_res or idx in self.donors_by_res):
                out.add(idx)
                continue

            # NAME:INDEX(:CHAIN) -> use INDEX
            parts = s_str.split(":")
            if len(parts) >= 2:
                idx2 = self._try_int(parts[1])
                if idx2 is not None and (idx2 in self.acceptors_by_res or idx2 in self.donors_by_res):
                    out.add(idx2)

        return sorted(out)

    @staticmethod
    def _try_int(x) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None
