"""
Utility script to extract Bruker SER files from a dataset tree.

Algorithm:
1) Traverse an input folder recursively and find every file named "ser".
2) Build a list where each element is a dict with two keys:
   - "name": identifier used by this program
   - "path": location of the SER file
3) For every dict element in that list:
   - map the SER to a matrix
   - export to .npy and/or .png based on export_npy and export_png
   - store the files in the output folder with proper name and extension
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nmrglue as ng
import numpy as np

# change directory to the location of this script
import os 
os.chdir(Path(__file__).parent)

def discover_ser_entries(input_folder: str | Path) -> list[dict[str, str]]:
	"""Return a list of dicts with SER paths and normalized names."""
	root = Path(input_folder).expanduser().resolve()
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Input folder does not exist or is not a directory: {root}")

	entries: list[dict[str, str]] = []
	for ser_path in sorted(root.rglob("ser")):
		if not ser_path.is_file():
			continue

		# Keep names stable and readable from nested paths.
		rel_parent = ser_path.parent.relative_to(root)
		name = str(rel_parent).replace("\\", "_").replace("/", "_").strip("_")
		if not name:
			name = "ser"

		entries.append({"name": name, "path": str(ser_path)})

	return entries


def _load_ser_matrix(ser_path: str | Path) -> np.ndarray:
    """
    Load a SER from disk and return it as a numpy array.

    nmrglue reads Bruker data from the experiment directory,
    so we pass the SER parent directory.
    """
    ser_file = Path(ser_path).expanduser().resolve()
    if ser_file.is_dir():
       bruker_dir = ser_file
    else:
       bruker_dir = ser_file.parent

    _dic, data = ng.bruker.read(str(bruker_dir))
    return np.asarray(data)


def _to_png_matrix(data: np.ndarray) -> np.ndarray:
	"""Convert complex/ND matrix to a 2D magnitude matrix suitable for PNG export."""
	mag = np.abs(np.asarray(data))
	mag = np.squeeze(mag)

	if mag.ndim == 0:
		return mag.reshape(1, 1)
	if mag.ndim == 1:
		return mag.reshape(1, -1)
	if mag.ndim == 2:
		return mag

	# For higher dimensions, collapse trailing axes into a 2D view.
	return mag.reshape(mag.shape[0], -1)


def export_ser_entries(
	ser_entries: list[dict[str, str]],
	output_folder: str | Path,
	export_png: bool = True,
	export_npy: bool = True,
) -> list[dict[str, Any]]:
	"""
	Process and export SER entries.

	Returns a report list with each file status.
	"""
	if not export_png and not export_npy:
		raise ValueError("At least one export format must be enabled.")

	out_dir = Path(output_folder).expanduser().resolve() / "SER_files"
	out_dir.mkdir(parents=True, exist_ok=True)

	report: list[dict[str, Any]] = []

	for entry in ser_entries:
		name = entry.get("name")
		ser_path = entry.get("path")

		if not name or not ser_path:
			report.append({"entry": entry, "status": "skipped", "reason": "missing name/path"})
			continue

		try:
			matrix = _load_ser_matrix(ser_path)
		except Exception as exc:
			report.append(
				{
					"name": name,
					"path": ser_path,
					"status": "error",
					"reason": f"failed to load SER: {exc}",
				}
			)
			continue

		exported_files: list[str] = []

		if export_npy:
			npy_path = out_dir / f"{name}.npy"
			np.save(npy_path, matrix)
			exported_files.append(str(npy_path))

		if export_png:
			# Lazy import keeps PNG dependency optional unless requested.
			import matplotlib.pyplot as plt

			png_path = out_dir / f"{name}.png"
			png_matrix = _to_png_matrix(matrix)
			plt.imsave(png_path, png_matrix, cmap="gray", origin="lower")
			exported_files.append(str(png_path))

		report.append(
			{
				"name": name,
				"path": ser_path,
				"status": "ok",
				"shape": tuple(matrix.shape),
				"exports": exported_files,
			}
		)

	return report


def extract_ser_files(
	input_folder: str | Path,
	output_folder: str | Path,
	export_png: bool = True,
	export_npy: bool = True,
) -> list[dict[str, Any]]:
	"""Discover SER files from input_folder and export them to output_folder/SER_files."""
	ser_entries = discover_ser_entries(input_folder)
	return export_ser_entries(
		ser_entries=ser_entries,
		output_folder=output_folder,
		export_png=export_png,
		export_npy=export_npy,
	)


if __name__ == "__main__":
	import argparse

	results = extract_ser_files(
		input_folder="../datasets",
		output_folder="../00_dataset",
		export_png=True,
		export_npy=True,
	)

	ok_count = sum(1 for r in results if r.get("status") == "ok")
	print(f"Processed {len(results)} SER entries, successful: {ok_count}")
	for item in results:
		if item.get("status") == "ok":
			print(f"[OK] {item['name']} -> {', '.join(item['exports'])}")
		else:
			print(f"[SKIP/ERR] {item}")
