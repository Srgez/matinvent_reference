import io
import os
import contextlib
from functools import lru_cache
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import lmdb
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential
from pymatgen.core.structure import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
import mattergen.evaluation.utils.lmdb_utils as lmdb_utils
import mattergen.evaluation.reference.reference_dataset_serializer as ref_serializer
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher

from rewards.calculators.base import Calculator


def _patch_lmdb_open_once() -> None:
    if getattr(lmdb_utils, "_matinvent_lmdb_patch_applied", False):
        return

    original_lmdb_open = lmdb_utils.lmdb_open

    def _patched_lmdb_open(db_path: str | os.PathLike, readonly: bool = False):
        if readonly:
            return lmdb.open(
                str(db_path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
        return original_lmdb_open(db_path, readonly=False)

    lmdb_utils.lmdb_open = _patched_lmdb_open
    # reference_dataset_serializer imports lmdb_open symbol directly; patch both.
    ref_serializer.lmdb_open = _patched_lmdb_open

    # Avoid a second lmdb.open() during LMDBBackedReferenceDatasetImpl init.
    # Upstream implementation calls lmdb_read_metadata(lmdb_path, ...), which re-opens
    # the same environment while self.env is already open in this process.
    def _patched_build_num_entries_by_chemsys_reduced_formulas(self, lmdb_path):
        result = defaultdict(dict)
        with self.env.begin() as txn:
            chemical_systems = lmdb_utils.lmdb_get(txn, "chemical_systems")
            for chemsys in chemical_systems:
                reduced_formulas = lmdb_utils.lmdb_get(txn, f"{chemsys}.reduced_formulas")
                for reduced_formula in reduced_formulas:
                    result[chemsys][reduced_formula] = lmdb_utils.lmdb_get(
                        txn, f"{chemsys}.{reduced_formula}.length"
                    )
        return {key: val for key, val in result.items()}

    ref_serializer.LMDBBackedReferenceDatasetImpl._build_num_entries_by_chemsys_reduced_formulas = (
        _patched_build_num_entries_by_chemsys_reduced_formulas
    )
    lmdb_utils._matinvent_lmdb_patch_applied = True


@lru_cache(maxsize=4)
def _load_reference_dataset(reference_dataset_path: str):
    _patch_lmdb_open_once()
    return LMDBGZSerializer().deserialize(reference_dataset_path)


@lru_cache(maxsize=4)
def _load_potential(device: str, potential_load_path: str):
    return Potential.from_checkpoint(
        device=device,
        load_path=potential_load_path,
        load_training_state=False,
    )


class MatterSim(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = "ehull",
        device: str | None = None,
        potential_load_path: str | None = None,
        reference_dataset_path: str | None = None,
        batch_size: int = 64,
        silent: bool = True,
    ) -> None:
        super().__init__(root_dir, task)
        self.device = device or "cpu"
        self.potential_load_path = potential_load_path
        self.reference_dataset_path = reference_dataset_path
        self.batch_size = batch_size
        self.silent = silent

    def _predict_total_energies(self, structures: List[Structure]) -> np.ndarray:
        atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
        dataloader = build_dataloader(
            atoms=atoms,
            batch_size=self.batch_size,
            shuffle=False,
            only_inference=True,
        )
        potential = _load_potential(self.device, self.potential_load_path)
        total_energy, _, _ = potential.predict_properties(
            dataloader,
            include_forces=False,
            include_stresses=False,
        )
        return np.array(total_energy, dtype=float)

    def _compute_ehull(self, structures: List[Structure], total_energies: np.ndarray) -> np.ndarray:
        reference = _load_reference_dataset(self.reference_dataset_path)
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=structures,
            energies=total_energies.tolist(),
            reference=reference,
            original_structures=structures,
            structure_matcher=DefaultDisorderedStructureMatcher(),
            energy_correction_scheme=MaterialsProject2020Compatibility(),
        )
        return np.array(evaluator.energy_capability.energy_above_hull, dtype=float)

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> np.ndarray:
        if self.task not in {"ehull", "energy_above_hull"}:
            raise ValueError(f"{self.task} is unknown task for MatterSim calculator!")
        if self.potential_load_path is None:
            raise ValueError("potential_load_path is required for MatterSim ehull calculation.")
        if self.reference_dataset_path is None:
            raise ValueError("reference_dataset_path is required for MatterSim ehull calculation.")

        structures = samples[0]
        out_path = os.path.abspath(os.path.join(self.root_dir, f"{label}.txt"))
        energies_path = os.path.abspath(os.path.join(self.root_dir, f"{label}_total_energy.txt"))

        if self.silent:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                total_energies = self._predict_total_energies(structures)
                results = self._compute_ehull(structures, total_energies)
        else:
            total_energies = self._predict_total_energies(structures)
            results = self._compute_ehull(structures, total_energies)

        np.savetxt(out_path, results, fmt="%.6f")
        np.savetxt(energies_path, total_energies, fmt="%.6f")
        return results
