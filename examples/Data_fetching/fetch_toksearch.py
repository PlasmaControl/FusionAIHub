import os
import subprocess
import sys
import time

import h5py
import numpy as np
from toksearch import MdsSignal, Pipeline, PtDataSignal
from tqdm import tqdm

# this one runs on iris, run the following
# module purge
# module load toksearch

# for copy:
# scp -r -o 'ProxyCommand ssh -p 2039 curiem@cybele.gat.com -W %h:%p' \
#   curiem@iris.gat.com:/cscratch/curiem/Data_fetch_TS/15* ./

# ***********start of user block******************
# limit of the size
size_GB = 400

# After fetching (interval) discharges, check the total directory size
interval = 100

# Root directory of the user for total size check
directory_path = "/cscratch/curiem"

# list of discharges to fetch
shots = list(np.arange(150000, 170000, dtype=int))

# shots = list(np.arange(175910,190000,dtype=int))
# one can set start_shot the where to start. (usually used for restarting the
# fetching due to unexpected termination)
start_shot = min(shots)

# path to save the files
path = "/cscratch/curiem/Data_fetch_CO2_s/"

# diag_names=[mag,mag_hi,bes,ece_cali,ece_s, co2_den, co2_pl, co2_s,
# ts,ts_rz,ts_error,custom]
diag_name = "co2_s"

# Define 3D diagnostics that have rhon data
diag_3d = ["ts", "ts_rz"]

# custom sig_names_custom, the suffix is fixed to be custom for now.
if diag_name == "custom":
    sig_names_custom = [""]
    names_custom = [""]
    tree = ""  # ptdata fo PTDATA, other trees names for MDS+
# ***********end of user block******************

shots.sort()


def size_limiter(
    directory_path: str = "/cscratch/curiem",
    size_GB: float = 450,
) -> None:
    try:
        size = (
            subprocess.check_output(["du", "-sh", directory_path])
            .split()[0]
            .decode("utf-8")
        )
    except subprocess.CalledProcessError as e:
        print(f"Error fetching directory size: {e}")
        return

    print(f"Size of {directory_path}: {size}")

    if size[-1] == "G" and float(size[:-1]) > size_GB:
        print(f"Size exceeds {size_GB}GB. Stopping...")
        sys.exit(1)


def size_limiter_sleep(
    directory_path: str = "/cscratch/curiem", size_GB: float = 450
) -> None:
    try:
        size = (
            subprocess.check_output(["du", "-sh", directory_path])
            .split()[0]
            .decode("utf-8")
        )
    except subprocess.CalledProcessError as e:
        print(f"Error fetching directory size: {e}")
        sys.exit(1)

    print(f"Size of {directory_path}: {size}")

    if size[-1] == "G" and float(size[:-1]) > size_GB:
        print(f"Size exceeds {size_GB}GB. Sleeping for 1hr...")

        # Sleep for 1 hour
        time.sleep(3600)  # 3600 seconds = 1 hour
        print("1 hour has passed. Checking size again...")

        try:
            size = (
                subprocess.check_output(["du", "-sh", directory_path])
                .split()[0]
                .decode("utf-8")
            )
        except subprocess.CalledProcessError as e:
            print(f"Error fetching directory size: {e}")
            sys.exit(1)

        if size[-1] == "G" and float(size[:-1]) > size_GB:
            print(f"Size still exceeds {size_GB}GB. Stoping")
            sys.exit(1)


def save_dict_to_hdf5(
    dictionary: dict,
    h5file: h5py.File,
) -> None:
    for key, value in dictionary.items():
        if isinstance(value, dict):
            group = h5file.create_group(key)
            save_dict_to_hdf5(value, group)
        else:
            h5file.create_dataset(key, data=value)


# generate the name and signal to fetch i ntoksearch
def signal_gen(
    diag_name: str = "zipfit",
    sig_names_custom: list | None = None,
    names_custom: list | None = None,
    tree_custom: str = "",
) -> tuple[list, list]:
    if names_custom is None:
        names_custom = [""]
    if sig_names_custom is None:
        sig_names_custom = [""]
    signals = []
    names = []

    # Counter: fncrate**
    # Adjustable scintillator: fplastic, fzns
    # Fixed scintillator: plasticfx*
    # Approximate calibrated signal: neutronsrate*
    if diag_name == "neutron":
        sig_names = [
            "fplastic",
            "fzns",
            "fncrate01",
            "fncrate02",
            "fncrate03",
            "fncrate04",
            "plasticfx1",
            "plasticfx2",
            "plasticfx3",
            "plasticfx4",
            "neutronsrate1",
            "neutronsrate2",
            "neutronsrate3",
            "neutronsrate4",
        ]
        names = [
            "fplastic",
            "fzns",
            "fncrate01",
            "fncrate02",
            "fncrate03",
            "fncrate04",
            "plasticfx1",
            "plasticfx2",
            "plasticfx3",
            "plasticfx4",
            "cali.neutronsrate1",
            "cali.neutronsrate2",
            "cali.neutronsrate3",
            "cali.neutronsrate4",
        ]

    elif diag_name == "mag_full":
        sig_name_without_d = [
            "mpi11m322",
            "mpi1a322",
            "mpi2a322",
            "mpi3a322",
            "mpi4a322",
            "mpi5a322",
            "mpi8a322",
            "mpi89a322",
            "mpi9a322",
            "mpi79fa322",
            "mpi79na322",
            "mpi7fa322",
            "mpi7na322",
            "mpi67a322",
            "mpi6fa322",
            "mpi6na322",
            "mpi66m322",
            "mpi1b322",
            "mpi2b322",
            "mpi3b322",
            "mpi4b322",
            "mpi5b322",
            "mpi8b322",
            "mpi89b322",
            "mpi9b322",
            "mpi79b322",
            "mpi7fb322",
            "mpi7nb322",
            "mpi67b322",
            "mpi6fb322",
            "mpi6nb322",
            "mpi2a067",
            "mpi11m067",
            "mpi2b067",
            "mpi67a097",
            "mpi67a067",
            "mpi66m067",
            "mpi67b097",
            "mpi67b067",
            "mpi1a139",
            "mpi2a139",
            "mpi3a139",
            "mpi4a139",
            "mpi5a139",
            "mpi79a147",
            "mpi67a142",
            "mpi67a157",
            "mpi6na132",
            "mpi6na157",
            "mpi66m157",
            "mpi6nb157",
            "mpi6fb142",
            "mpi67b157",
            "mpi7nb142",
            "mpi79b142",
            "mpi5b139",
            "mpi4b139",
            "mpi3b139",
            "mpi2b139",
            "mpi1b139",
            "mpi1b157",
            "mpi1u157",
            "mpi2u157",
            "mpi3u157",
            "mpi4u157",
            "mpi5u157",
            "mpi6u157",
            "mpi7u157",
            "dsl1u180",
            "dsl2u180",
            "dsl3u180",
            "dsl4u157",
            "dsl5u157",
            "dsl6u157",
            "mpi66m127",
            "mpi66m132",
            "mpi66m137",
            "mpi66b137",
            "mpi6nb137",
            "mpi66m307",
            "mpi66m312",
            "mpi6na312",
            "mpi66b312",
            "mpi6nb312",
            "mpi66m322",
            "mpi1l020",
            "mpi2l020",
            "mpi1l050",
            "mpi1l110",
            "mpi1l180",
            "mpi2l180",
            "mpi3l180",
            "mpi1l230",
            "mpi1l320",
            "mpi66m020",
            "mpi66m067",
            "mpi66m097",
            "mpi66m127",
            "mpi66m132",
            "mpi66m137",
            "mpi66m157",
            "mpi66m200",
            "mpi66m247",
            "mpi66m277",
            "mpi66m307",
            "mpi66m312",
            "mpi66m322",
            "mpi66m340",
            "mpi67a022",
            "mpi67a037",
            "mpi67a1",
            "mpi67a052",
            "mpi67a067",
            "mpi67a082",
            "mpi67a097",
            "mpi67a2",
            "mpi67a142",
            "mpi67a157",
            "mpi67a3",
            "mpi67a217",
            "mpi67a4",
            "mpi67a262",
            "mpi67a277",
            "mpi67a5",
            "mpi67a307",
            "mpi67a337",
            "mpi67a6",
            "mpi67b022",
            "mpi67b037",
            "mpi67b1",
            "mpi67b052",
            "mpi67b097",
            "mpi67b2",
            "mpi67b157",
            "mpi67b3",
            "mpi67b217",
            "mpi67b4",
            "mpi67b277",
            "mpi67b5",
            "mpi67b337",
            "mpi67b6",
            "mpi79a072",
            "mpi79a147",
            "mpi79a222",
            "mpi79a272",
            "mpi79b067",
            "mpi79b142",
            "mpi79b217",
            "mpi79b277",
            "mpi5a139",
            "mpi4a139",
            "mpi3a139",
            "mpi2a139",
            "mpi1a139",
            "mpi1b139",
            "mpi2b139",
            "mpi3b139",
            "mpi4b139",
            "mpi5b139",
            "mpi5a199",
            "mpi4a199",
            "mpi3a199",
            "mpi2a199",
            "mpi1a199",
            "mpi1b199",
            "mpi2b199",
            "mpi3b199",
            "mpi4b199",
            "mpi5b199",
            "mpi1a011",
            "mpi1a049",
            "mpi1a109",
            "mpi1a139",
            "mpi1a199",
            "mpi1a244",
            "mpi1a274",
            "mpi1a341",
            "mpi1b011",
            "mpi1b049",
            "mpi1b109",
            "mpi1b139",
            "mpi1b199",
            "mpi1b244",
            "mpi1b274",
            "mpi1b341",
            "isl66m017",
            "isl66m042",
            "isl66m072",
            "isl66m102",
            "isl66m132",
            "isl66m197",
            "isl66m252",
            "isl66m312",
            "isl67a017",
            "isl67a052",
            "isl67a072",
            "isl67a112",
            "isl67a132",
            "isl67a197",
            "isl67a252",
            "isl67a312",
            "isl67b017",
            "isl67b052",
            "isl67b072",
            "isl67b112",
            "isl67b132",
            "isl67b197",
            "isl67b252",
            "isl67b312",
            "isl79a072",
            "isl79a147",
            "isl79a222",
            "isl79a272",
            "isl79b067",
            "isl79b142",
            "isl79b217",
            "isl79b277",
            "isl5a139",
            "isl4a139",
            "isl3a139",
            "isl2a139",
            "isl1a139",
            "isl1b139",
            "isl2b139",
            "isl3b139",
            "isl4b139",
            "isl5b139",
            "isl5a199",
            "isl4a199",
            "isl3a199",
            "isl2a199",
            "isl1a199",
            "isl1b199",
            "isl2b199",
            "isl3b199",
            "isl4b199",
            "isl5b199",
            "isl1a011",
            "isl1a049",
            "isl1a109",
            "isl1a139",
            "isl1a199",
            "isl1a244",
            "isl1a274",
            "isl1a341",
            "isl1b011",
            "isl1b049",
            "isl1b109",
            "isl1b139",
            "isl1b199",
            "isl1b244",
            "isl1b274",
            "isl1b341",
            "dsl12a067",
            "dsl34a067",
            "dsl59a067",
            "dsl79a067",
            "dsl67a067",
            "dsl66m052",
            "dsl67b067",
            "dsl79b067",
            "dsl59b067",
            "dsl34b067",
            "dsl12b067",
            "dsl12a157",
            "dsl34a157",
            "dsl59a157",
            "dsl79a157",
            "dsl67a157",
            "dsl66m152",
            "dsl67b157",
            "dsl79b157",
            "dsl59b157",
            "dsl34b157",
            "dsl12b157",
            "dsl67a067",
            "dsl67a157",
            "sl67fa345",
            "sl67na345",
            "dsl66m052",
            "sl66a132",
            "sl66b132",
            "dsl66m152",
            "sl66a312",
            "sl66b312",
            "sl67nb015",
            "sl67fb015",
            "dsl67b067",
            "dsl67b157",
            "esl66m019",
            "esl019",
            "esl66m079",
            "esl079",
            "esl66m139",
            "esl139",
            "esl66m199",
            "esl199",
            "esl66m259",
            "esl259",
            "esl66m319",
            "esl319",
            "esl67a004",
            "esl67a034",
            "esl67a064",
            "esl67a094",
            "esl67a124",
            "esl67a154",
            "esl67a184",
            "esl67a214",
            "esl67a244",
            "esl67a274",
            "esl67a304",
            "esl67a334",
            "esl67b004",
            "esl67b034",
            "esl67b064",
            "esl67b094",
            "esl67b124",
            "esl67b154",
            "esl67b184",
            "esl67b214",
            "esl67b244",
            "esl67b274",
            "esl67b304",
            "esl67b334",
            "bti66m053",
            "bti66m132",
            "bti66m233",
            "bti66m312",
            "psf1a",
            "psf1a",
            "psf1a",
            "psf1a",
            "psf6natotl",
            "psf6na",
            "psi11mtotl",
            "psi11m",
            "psi6atotl",
            "psi6a",
            "psf1a",
            "psf6natotl",
            "psi11mtotl",
            "psi6atotl",
            "psf2a",
            "psf3a",
            "psf4a",
            "psf5a",
            "psf8a",
            "psf9a",
            "psf7fa",
            "psf7na",
            "psf6fa",
            "psf6na",
            "psf6nb",
            "psf6fb",
            "psf7nb",
            "psf7fb",
            "psf9b",
            "psf8b",
            "psf5b",
            "psf4b",
            "psf3b",
            "psf2b",
            "psf1b",
            "psi11m",
            "psi12a",
            "psi23a",
            "psi34a",
            "psi45a",
            "psi58a",
            "psi9a",
            "psi7a",
            "psi6a",
            "psi6b",
            "psi7b",
            "psi9b",
            "psi89nb",
            "psi89fb",
            "psi58b",
            "psi45b",
            "psi34b",
            "psi23b",
            "psi12b",
            "psi1l",
            "psi2l",
            "psi3l",
            "mpi1b",
            "mpi66m020",
            "mpi66m097",
            "mpi66m020",
            "mpi66m097",
            "mpi66m067",
            "mpi66m247",
            "mpi66m097",
            "mpi66m277",
            "mpi66m127",
            "mpi66m307",
            "mpi66m157",
            "mpi66m340",
            "mpi66m200",
            "mpi66m020",
            "mpi66m247",
            "mpi66m127",
            "mpi66m277",
            "mpi66m157",
            "mpi66m307",
            "mpi66m200",
            "mpi66m340",
            "mpi66m067",
            "mpi67a022",
            "mpi67a217",
            "mpi67a037",
            "mpi67a067",
            "mpi67a052",
            "mpi67a022",
            "mpi67a067",
            "mpi67a262",
            "mpi67a082",
            "mpi67a052",
            "mpi67a097",
            "mpi67a082",
            "mpi67a142",
            "mpi67a037",
            "mpi67a217",
            "mpi67a097",
            "mpi67a262",
            "mpi67a277",
            "mpi67a277",
            "mpi67a307",
            "mpi67a307",
            "mpi67a337",
            "mpi67a337",
            "mpi67a142",
            "mpi67b022",
            "mpi67b052",
            "mpi67b037",
            "mpi67b217",
            "mpi67b052",
            "mpi67b037",
            "mpi67b097",
            "mpi67b277",
            "mpi67b157",
            "mpi67b337",
            "mpi67b217",
            "mpi67b097",
            "mpi67b277",
            "mpi67b157",
            "mpi67b337",
            "mpi67b022",
            "mpi79a072",
            "mpi79a222",
            "mpi79a147",
            "mpi79a072",
            "mpi79a222",
            "mpi79a272",
            "mpi79a272",
            "mpi79a147",
            "mpi79b067",
            "mpi79b217",
            "mpi79b142",
            "mpi79b067",
            "mpi79b217",
            "mpi79b277",
            "mpi79b277",
            "mpi79b142",
            "mpi1a011",
            "mpi1a199",
            "mpi1a049",
            "mpi1a244",
            "mpi1a109",
            "mpi1a011",
            "mpi1a139",
            "mpi1a341",
            "mpi1a199",
            "mpi1a139",
            "mpi1a244",
            "mpi1a274",
            "mpi1a274",
            "mpi1a109",
            "mpi1a341",
            "mpi1a049",
            "mpi1b011",
            "mpi1b199",
            "mpi1b049",
            "mpi1b244",
            "mpi1b109",
            "mpi1b011",
            "mpi1b139",
            "mpi1b341",
            "mpi1b199",
            "mpi1b139",
            "mpi1b244",
            "mpi1b274",
            "mpi1b274",
            "mpi1b109",
            "mpi1b341",
            "mpi1b049",
            "mpi5a199",
            "mpi5a139",
            "mpi4a199",
            "mpi4a139",
            "mpi3a199",
            "mpi3a139",
            "mpi2a199",
            "mpi2a139",
            "mpi1a199",
            "mpi1a139",
            "mpi1b199",
            "mpi1b139",
            "mpi2b199",
            "mpi2b139",
            "mpi3b199",
            "mpi3b139",
            "mpi4b199",
            "mpi4b139",
            "mpi5b199",
            "mpi5b139",
            "isl66m017",
            "isl66m042",
            "isl66m042",
            "isl66m072",
            "isl66m072",
            "isl66m252",
            "isl66m102",
            "isl66m132",
            "isl66m132",
            "isl66m312",
            "isl66m197",
            "isl66m017",
            "isl66m252",
            "isl66m102",
            "isl66m312",
            "isl66m197",
            "isl67a017",
            "isl67a052",
            "isl67a052",
            "isl67a072",
            "isl67a072",
            "isl67a252",
            "isl67a112",
            "isl67a132",
            "isl67a132",
            "isl67a312",
            "isl67a197",
            "isl67a017",
            "isl67a252",
            "isl67a112",
            "isl67a312",
            "isl67a197",
            "isl67b017",
            "isl67b052",
            "isl67b052",
            "isl67b072",
            "isl67b072",
            "isl67b252",
            "isl67b112",
            "isl67b132",
            "isl67b132",
            "isl67b312",
            "isl67b197",
            "isl67b017",
            "isl67b252",
            "isl67b112",
            "isl67b312",
            "isl67b197",
            "isl79a072",
            "isl79a222",
            "isl79a147",
            "isl79a072",
            "isl79a222",
            "isl79a272",
            "isl79a272",
            "isl79a147",
            "isl79b067",
            "isl79b217",
            "isl79b142",
            "isl79b067",
            "isl79b217",
            "isl79b277",
            "isl79b277",
            "isl79b142",
            "isl1a011",
            "isl1a199",
            "isl1a049",
            "isl1a244",
            "isl1a109",
            "isl1a011",
            "isl1a139",
            "isl1a341",
            "isl1a199",
            "isl1a139",
            "isl1a244",
            "isl1a274",
            "isl1a274",
            "isl1a109",
            "isl1a341",
            "isl1a049",
            "isl1b011",
            "isl1b199",
            "isl1b049",
            "isl1b244",
            "isl1b109",
            "isl1b011",
            "isl1b139",
            "isl1b341",
            "isl1b199",
            "isl1b139",
            "isl1b244",
            "isl1b274",
            "isl1b274",
            "isl1b109",
            "isl1b341",
            "isl1b049",
            "isl5a199",
            "isl5a139",
            "isl4a199",
            "isl4a139",
            "isl3a199",
            "isl3a139",
            "isl2a199",
            "isl2a139",
            "isl1a199",
            "isl1a139",
            "isl1b199",
            "isl1b139",
            "isl2b199",
            "isl2b139",
            "isl3b199",
            "isl3b139",
            "isl4b199",
            "isl4b139",
            "isl5b199",
            "isl5b139",
            "esl66m019",
            "esl66m079",
            "esl66m079",
            "esl66m259",
            "esl66m139",
            "esl66m319",
            "esl66m199",
            "esl66m019",
            "esl66m259",
            "esl66m139",
            "esl66m319",
            "esl66m199",
            "esl66m079",
            "esl66m259",
            "esl66m139",
            "esl66m319",
            "esl66m199",
            "esl66m019",
            "esl67a004",
            "esl67a244",
            "esl67a034",
            "esl67a154",
            "esl67a064",
            "esl67a184",
            "esl67a094",
            "esl67a274",
            "esl67a124",
            "esl67a304",
            "esl67a154",
            "esl67a334",
            "esl67a184",
            "esl67a004",
            "esl67a214",
            "esl67a094",
            "esl67a244",
            "esl67a124",
            "esl67a274",
            "esl67a034",
            "esl67a304",
            "esl67a064",
            "esl67a334",
            "esl67a214",
            "esl67b004",
            "esl67b244",
            "esl67b034",
            "esl67b154",
            "esl67b064",
            "esl67b184",
            "esl67b094",
            "esl67b274",
            "esl67b124",
            "esl67b304",
            "esl67b154",
            "esl67b334",
            "esl67b184",
            "esl67b004",
            "esl67b214",
            "esl67b094",
            "esl67b244",
            "esl67b124",
            "esl67b274",
            "esl67b034",
            "esl67b304",
            "esl67b064",
            "esl67b334",
            "esl67b214",
            "bti66m053",
            "bti66m233",
            "bti66m132",
            "bti66m053",
            "bti66m233",
            "bti66m312",
            "bti66m312",
            "bti66m132",
            "mpi2a067",
            "mpi1u157",
            "isl79a",
        ]

        name_without_d = [
            "mpi.11.m.322",
            "mpi.1.a.322",
            "mpi.2.a.322",
            "mpi.3.a.322",
            "mpi.4.a.322",
            "mpi.5.a.322",
            "mpi.8.a.322",
            "mpi.89.a.322",
            "mpi.9.a.322",
            "mpi.79.fa.322",
            "mpi.79.na.322",
            "mpi.7.fa.322",
            "mpi.7.na.322",
            "mpi.67.a.322",
            "mpi.6.fa.322",
            "mpi.6.na.322",
            "mpi.66.m.322",
            "mpi.1.b.322",
            "mpi.2.b.322",
            "mpi.3.b.322",
            "mpi.4.b.322",
            "mpi.5.b.322",
            "mpi.8.b.322",
            "mpi.89.b.322",
            "mpi.9.b.322",
            "mpi.79.b.322",
            "mpi.7.fb.322",
            "mpi.7.nb.322",
            "mpi.67.b.322",
            "mpi.6.fb.322",
            "mpi.6.nb.322",
            "mpi.2.a.067",
            "mpi.11.m.067",
            "mpi.2.b.067",
            "mpi.67.a.097",
            "mpi.67.a.067",
            "mpi.66.m.067",
            "mpi.67.b.097",
            "mpi.67.b.067",
            "mpi.1.a.139",
            "mpi.2.a.139",
            "mpi.3.a.139",
            "mpi.4.a.139",
            "mpi.5.a.139",
            "mpi.79.a.147",
            "mpi.67.a.142",
            "mpi.67.a.157",
            "mpi.6.na.132",
            "mpi.6.na.157",
            "mpi.66.m.157",
            "mpi.6.nb.157",
            "mpi.6.fb.142",
            "mpi.67.b.157",
            "mpi.7.nb.142",
            "mpi.79.b.142",
            "mpi.5.b.139",
            "mpi.4.b.139",
            "mpi.3.b.139",
            "mpi.2.b.139",
            "mpi.1.b.139",
            "mpi.1.b.157",
            "mpi.1.u.157",
            "mpi.2.u.157",
            "mpi.3.u.157",
            "mpi.4.u.157",
            "mpi.5.u.157",
            "mpi.6.u.157",
            "mpi.7.u.157",
            "dsl.1.u.180",
            "dsl.2.u.180",
            "dsl.3.u.180",
            "dsl.4.u.157",
            "dsl.5.u.157",
            "dsl.6.u.157",
            "mpi.66.m.127",
            "mpi.66.m.132",
            "mpi.66.m.137",
            "mpi.66.b.137",
            "mpi.6.nb.137",
            "mpi.66.m.307",
            "mpi.66.m.312",
            "mpi.6.na.312",
            "mpi.66.b.312",
            "mpi.6.nb.312",
            "mpi.66.m.322",
            "mpi.1.l.020",
            "mpi.2.l.020",
            "mpi.1.l.050",
            "mpi.1.l.110",
            "mpi.1.l.180",
            "mpi.2.l.180",
            "mpi.3.l.180",
            "mpi.1.l.230",
            "mpi.1.l.320",
            "mpi.66.m.020",
            "mpi.66.m.067",
            "mpi.66.m.097",
            "mpi.66.m.127",
            "mpi.66.m.132",
            "mpi.66.m.137",
            "mpi.66.m.157",
            "mpi.66.m.200",
            "mpi.66.m.247",
            "mpi.66.m.277",
            "mpi.66.m.307",
            "mpi.66.m.312",
            "mpi.66.m.322",
            "mpi.66.m.340",
            "mpi.67.a.022",
            "mpi.67.a.037",
            "mpi.67.a.1",
            "mpi.67.a.052",
            "mpi.67.a.067",
            "mpi.67.a.082",
            "mpi.67.a.097",
            "mpi.67.a.2",
            "mpi.67.a.142",
            "mpi.67.a.157",
            "mpi.67.a.3",
            "mpi.67.a.217",
            "mpi.67.a.4",
            "mpi.67.a.262",
            "mpi.67.a.277",
            "mpi.67.a.5",
            "mpi.67.a.307",
            "mpi.67.a.337",
            "mpi.67.a.6",
            "mpi.67.b.022",
            "mpi.67.b.037",
            "mpi.67.b.1",
            "mpi.67.b.052",
            "mpi.67.b.097",
            "mpi.67.b.2",
            "mpi.67.b.157",
            "mpi.67.b.3",
            "mpi.67.b.217",
            "mpi.67.b.4",
            "mpi.67.b.277",
            "mpi.67.b.5",
            "mpi.67.b.337",
            "mpi.67.b.6",
            "mpi.79.a.072",
            "mpi.79.a.147",
            "mpi.79.a.222",
            "mpi.79.a.272",
            "mpi.79.b.067",
            "mpi.79.b.142",
            "mpi.79.b.217",
            "mpi.79.b.277",
            "mpi.5.a.139",
            "mpi.4.a.139",
            "mpi.3.a.139",
            "mpi.2.a.139",
            "mpi.1.a.139",
            "mpi.1.b.139",
            "mpi.2.b.139",
            "mpi.3.b.139",
            "mpi.4.b.139",
            "mpi.5.b.139",
            "mpi.5.a.199",
            "mpi.4.a.199",
            "mpi.3.a.199",
            "mpi.2.a.199",
            "mpi.1.a.199",
            "mpi.1.b.199",
            "mpi.2.b.199",
            "mpi.3.b.199",
            "mpi.4.b.199",
            "mpi.5.b.199",
            "mpi.1.a.011",
            "mpi.1.a.049",
            "mpi.1.a.109",
            "mpi.1.a.139",
            "mpi.1.a.199",
            "mpi.1.a.244",
            "mpi.1.a.274",
            "mpi.1.a.341",
            "mpi.1.b.011",
            "mpi.1.b.049",
            "mpi.1.b.109",
            "mpi.1.b.139",
            "mpi.1.b.199",
            "mpi.1.b.244",
            "mpi.1.b.274",
            "mpi.1.b.341",
            "isl.66.m.017",
            "isl.66.m.042",
            "isl.66.m.072",
            "isl.66.m.102",
            "isl.66.m.132",
            "isl.66.m.197",
            "isl.66.m.252",
            "isl.66.m.312",
            "isl.67.a.017",
            "isl.67.a.052",
            "isl.67.a.072",
            "isl.67.a.112",
            "isl.67.a.132",
            "isl.67.a.197",
            "isl.67.a.252",
            "isl.67.a.312",
            "isl.67.b.017",
            "isl.67.b.052",
            "isl.67.b.072",
            "isl.67.b.112",
            "isl.67.b.132",
            "isl.67.b.197",
            "isl.67.b.252",
            "isl.67.b.312",
            "isl.79.a.072",
            "isl.79.a.147",
            "isl.79.a.222",
            "isl.79.a.272",
            "isl.79.b.067",
            "isl.79.b.142",
            "isl.79.b.217",
            "isl.79.b.277",
            "isl.5.a.139",
            "isl.4.a.139",
            "isl.3.a.139",
            "isl.2.a.139",
            "isl.1.a.139",
            "isl.1.b.139",
            "isl.2.b.139",
            "isl.3.b.139",
            "isl.4.b.139",
            "isl.5.b.139",
            "isl.5.a.199",
            "isl.4.a.199",
            "isl.3.a.199",
            "isl.2.a.199",
            "isl.1.a.199",
            "isl.1.b.199",
            "isl.2.b.199",
            "isl.3.b.199",
            "isl.4.b.199",
            "isl.5.b.199",
            "isl.1.a.011",
            "isl.1.a.049",
            "isl.1.a.109",
            "isl.1.a.139",
            "isl.1.a.199",
            "isl.1.a.244",
            "isl.1.a.274",
            "isl.1.a.341",
            "isl.1.b.011",
            "isl.1.b.049",
            "isl.1.b.109",
            "isl.1.b.139",
            "isl.1.b.199",
            "isl.1.b.244",
            "isl.1.b.274",
            "isl.1.b.341",
            "dsl.12.a.067",
            "dsl.34.a.067",
            "dsl.59.a.067",
            "dsl.79.a.067",
            "dsl.67.a.067",
            "dsl.66.m.052",
            "dsl.67.b.067",
            "dsl.79.b.067",
            "dsl.59.b.067",
            "dsl.34.b.067",
            "dsl.12.b.067",
            "dsl.12.a.157",
            "dsl.34.a.157",
            "dsl.59.a.157",
            "dsl.79.a.157",
            "dsl.67.a.157",
            "dsl.66.m.152",
            "dsl.67.b.157",
            "dsl.79.b.157",
            "dsl.59.b.157",
            "dsl.34.b.157",
            "dsl.12.b.157",
            "dsl.67.a.067",
            "dsl.67.a.157",
            "sl.67.fa.345",
            "sl.67.na.345",
            "dsl.66.m.052",
            "sl.66.a.132",
            "sl.66.b.132",
            "dsl.66.m.152",
            "sl.66.a.312",
            "sl.66.b.312",
            "sl.67.nb.015",
            "sl.67.fb.015",
            "dsl.67.b.067",
            "dsl.67.b.157",
            "esl.66.m.019",
            "esl.019..",
            "esl.66.m.079",
            "esl.079..",
            "esl.66.m.139",
            "esl.139..",
            "esl.66.m.199",
            "esl.199..",
            "esl.66.m.259",
            "esl.259..",
            "esl.66.m.319",
            "esl.319..",
            "esl.67.a.004",
            "esl.67.a.034",
            "esl.67.a.064",
            "esl.67.a.094",
            "esl.67.a.124",
            "esl.67.a.154",
            "esl.67.a.184",
            "esl.67.a.214",
            "esl.67.a.244",
            "esl.67.a.274",
            "esl.67.a.304",
            "esl.67.a.334",
            "esl.67.b.004",
            "esl.67.b.034",
            "esl.67.b.064",
            "esl.67.b.094",
            "esl.67.b.124",
            "esl.67.b.154",
            "esl.67.b.184",
            "esl.67.b.214",
            "esl.67.b.244",
            "esl.67.b.274",
            "esl.67.b.304",
            "esl.67.b.334",
            "bti.66.m.053",
            "bti.66.m.132",
            "bti.66.m.233",
            "bti.66.m.312",
            "psf.1.a.",
            "psf.1.a.",
            "psf.1.a.",
            "psf.1.a.",
            "psf.6.natotl.",
            "psf.6.na.",
            "psi.11.mtotl.",
            "psi.11.m.",
            "psi.6.atotl.",
            "psi.6.a.",
            "psf.1.a.",
            "psf.6.natotl.",
            "psi.11.mtotl.",
            "psi.6.atotl.",
            "psf.2.a.",
            "psf.3.a.",
            "psf.4.a.",
            "psf.5.a.",
            "psf.8.a.",
            "psf.9.a.",
            "psf.7.fa.",
            "psf.7.na.",
            "psf.6.fa.",
            "psf.6.na.",
            "psf.6.nb.",
            "psf.6.fb.",
            "psf.7.nb.",
            "psf.7.fb.",
            "psf.9.b.",
            "psf.8.b.",
            "psf.5.b.",
            "psf.4.b.",
            "psf.3.b.",
            "psf.2.b.",
            "psf.1.b.",
            "psi.11.m.",
            "psi.12.a.",
            "psi.23.a.",
            "psi.34.a.",
            "psi.45.a.",
            "psi.58.a.",
            "psi.9.a.",
            "psi.7.a.",
            "psi.6.a.",
            "psi.6.b.",
            "psi.7.b.",
            "psi.9.b.",
            "psi.89.nb.",
            "psi.89.fb.",
            "psi.58.b.",
            "psi.45.b.",
            "psi.34.b.",
            "psi.23.b.",
            "psi.12.b.",
            "psi.1.l.",
            "psi.2.l.",
            "psi.3.l.",
            "mpi.1.b.",
            "mpi.66.m.020",
            "mpi.66.m.097",
            "mpi.66.m.020",
            "mpi.66.m.097",
            "mpi.66.m.067",
            "mpi.66.m.247",
            "mpi.66.m.097",
            "mpi.66.m.277",
            "mpi.66.m.127",
            "mpi.66.m.307",
            "mpi.66.m.157",
            "mpi.66.m.340",
            "mpi.66.m.200",
            "mpi.66.m.020",
            "mpi.66.m.247",
            "mpi.66.m.127",
            "mpi.66.m.277",
            "mpi.66.m.157",
            "mpi.66.m.307",
            "mpi.66.m.200",
            "mpi.66.m.340",
            "mpi.66.m.067",
            "mpi.67.a.022",
            "mpi.67.a.217",
            "mpi.67.a.037",
            "mpi.67.a.067",
            "mpi.67.a.052",
            "mpi.67.a.022",
            "mpi.67.a.067",
            "mpi.67.a.262",
            "mpi.67.a.082",
            "mpi.67.a.052",
            "mpi.67.a.097",
            "mpi.67.a.082",
            "mpi.67.a.142",
            "mpi.67.a.037",
            "mpi.67.a.217",
            "mpi.67.a.097",
            "mpi.67.a.262",
            "mpi.67.a.277",
            "mpi.67.a.277",
            "mpi.67.a.307",
            "mpi.67.a.307",
            "mpi.67.a.337",
            "mpi.67.a.337",
            "mpi.67.a.142",
            "mpi.67.b.022",
            "mpi.67.b.052",
            "mpi.67.b.037",
            "mpi.67.b.217",
            "mpi.67.b.052",
            "mpi.67.b.037",
            "mpi.67.b.097",
            "mpi.67.b.277",
            "mpi.67.b.157",
            "mpi.67.b.337",
            "mpi.67.b.217",
            "mpi.67.b.097",
            "mpi.67.b.277",
            "mpi.67.b.157",
            "mpi.67.b.337",
            "mpi.67.b.022",
            "mpi.79.a.072",
            "mpi.79.a.222",
            "mpi.79.a.147",
            "mpi.79.a.072",
            "mpi.79.a.222",
            "mpi.79.a.272",
            "mpi.79.a.272",
            "mpi.79.a.147",
            "mpi.79.b.067",
            "mpi.79.b.217",
            "mpi.79.b.142",
            "mpi.79.b.067",
            "mpi.79.b.217",
            "mpi.79.b.277",
            "mpi.79.b.277",
            "mpi.79.b.142",
            "mpi.1.a.011",
            "mpi.1.a.199",
            "mpi.1.a.049",
            "mpi.1.a.244",
            "mpi.1.a.109",
            "mpi.1.a.011",
            "mpi.1.a.139",
            "mpi.1.a.341",
            "mpi.1.a.199",
            "mpi.1.a.139",
            "mpi.1.a.244",
            "mpi.1.a.274",
            "mpi.1.a.274",
            "mpi.1.a.109",
            "mpi.1.a.341",
            "mpi.1.a.049",
            "mpi.1.b.011",
            "mpi.1.b.199",
            "mpi.1.b.049",
            "mpi.1.b.244",
            "mpi.1.b.109",
            "mpi.1.b.011",
            "mpi.1.b.139",
            "mpi.1.b.341",
            "mpi.1.b.199",
            "mpi.1.b.139",
            "mpi.1.b.244",
            "mpi.1.b.274",
            "mpi.1.b.274",
            "mpi.1.b.109",
            "mpi.1.b.341",
            "mpi.1.b.049",
            "mpi.5.a.199",
            "mpi.5.a.139",
            "mpi.4.a.199",
            "mpi.4.a.139",
            "mpi.3.a.199",
            "mpi.3.a.139",
            "mpi.2.a.199",
            "mpi.2.a.139",
            "mpi.1.a.199",
            "mpi.1.a.139",
            "mpi.1.b.199",
            "mpi.1.b.139",
            "mpi.2.b.199",
            "mpi.2.b.139",
            "mpi.3.b.199",
            "mpi.3.b.139",
            "mpi.4.b.199",
            "mpi.4.b.139",
            "mpi.5.b.199",
            "mpi.5.b.139",
            "isl.66.m.017",
            "isl.66.m.042",
            "isl.66.m.042",
            "isl.66.m.072",
            "isl.66.m.072",
            "isl.66.m.252",
            "isl.66.m.102",
            "isl.66.m.132",
            "isl.66.m.132",
            "isl.66.m.312",
            "isl.66.m.197",
            "isl.66.m.017",
            "isl.66.m.252",
            "isl.66.m.102",
            "isl.66.m.312",
            "isl.66.m.197",
            "isl.67.a.017",
            "isl.67.a.052",
            "isl.67.a.052",
            "isl.67.a.072",
            "isl.67.a.072",
            "isl.67.a.252",
            "isl.67.a.112",
            "isl.67.a.132",
            "isl.67.a.132",
            "isl.67.a.312",
            "isl.67.a.197",
            "isl.67.a.017",
            "isl.67.a.252",
            "isl.67.a.112",
            "isl.67.a.312",
            "isl.67.a.197",
            "isl.67.b.017",
            "isl.67.b.052",
            "isl.67.b.052",
            "isl.67.b.072",
            "isl.67.b.072",
            "isl.67.b.252",
            "isl.67.b.112",
            "isl.67.b.132",
            "isl.67.b.132",
            "isl.67.b.312",
            "isl.67.b.197",
            "isl.67.b.017",
            "isl.67.b.252",
            "isl.67.b.112",
            "isl.67.b.312",
            "isl.67.b.197",
            "isl.79.a.072",
            "isl.79.a.222",
            "isl.79.a.147",
            "isl.79.a.072",
            "isl.79.a.222",
            "isl.79.a.272",
            "isl.79.a.272",
            "isl.79.a.147",
            "isl.79.b.067",
            "isl.79.b.217",
            "isl.79.b.142",
            "isl.79.b.067",
            "isl.79.b.217",
            "isl.79.b.277",
            "isl.79.b.277",
            "isl.79.b.142",
            "isl.1.a.011",
            "isl.1.a.199",
            "isl.1.a.049",
            "isl.1.a.244",
            "isl.1.a.109",
            "isl.1.a.011",
            "isl.1.a.139",
            "isl.1.a.341",
            "isl.1.a.199",
            "isl.1.a.139",
            "isl.1.a.244",
            "isl.1.a.274",
            "isl.1.a.274",
            "isl.1.a.109",
            "isl.1.a.341",
            "isl.1.a.049",
            "isl.1.b.011",
            "isl.1.b.199",
            "isl.1.b.049",
            "isl.1.b.244",
            "isl.1.b.109",
            "isl.1.b.011",
            "isl.1.b.139",
            "isl.1.b.341",
            "isl.1.b.199",
            "isl.1.b.139",
            "isl.1.b.244",
            "isl.1.b.274",
            "isl.1.b.274",
            "isl.1.b.109",
            "isl.1.b.341",
            "isl.1.b.049",
            "isl.5.a.199",
            "isl.5.a.139",
            "isl.4.a.199",
            "isl.4.a.139",
            "isl.3.a.199",
            "isl.3.a.139",
            "isl.2.a.199",
            "isl.2.a.139",
            "isl.1.a.199",
            "isl.1.a.139",
            "isl.1.b.199",
            "isl.1.b.139",
            "isl.2.b.199",
            "isl.2.b.139",
            "isl.3.b.199",
            "isl.3.b.139",
            "isl.4.b.199",
            "isl.4.b.139",
            "isl.5.b.199",
            "isl.5.b.139",
            "esl.66.m.019",
            "esl.66.m.079",
            "esl.66.m.079",
            "esl.66.m.259",
            "esl.66.m.139",
            "esl.66.m.319",
            "esl.66.m.199",
            "esl.66.m.019",
            "esl.66.m.259",
            "esl.66.m.139",
            "esl.66.m.319",
            "esl.66.m.199",
            "esl.66.m.079",
            "esl.66.m.259",
            "esl.66.m.139",
            "esl.66.m.319",
            "esl.66.m.199",
            "esl.66.m.019",
            "esl.67.a.004",
            "esl.67.a.244",
            "esl.67.a.034",
            "esl.67.a.154",
            "esl.67.a.064",
            "esl.67.a.184",
            "esl.67.a.094",
            "esl.67.a.274",
            "esl.67.a.124",
            "esl.67.a.304",
            "esl.67.a.154",
            "esl.67.a.334",
            "esl.67.a.184",
            "esl.67.a.004",
            "esl.67.a.214",
            "esl.67.a.094",
            "esl.67.a.244",
            "esl.67.a.124",
            "esl.67.a.274",
            "esl.67.a.034",
            "esl.67.a.304",
            "esl.67.a.064",
            "esl.67.a.334",
            "esl.67.a.214",
            "esl.67.b.004",
            "esl.67.b.244",
            "esl.67.b.034",
            "esl.67.b.154",
            "esl.67.b.064",
            "esl.67.b.184",
            "esl.67.b.094",
            "esl.67.b.274",
            "esl.67.b.124",
            "esl.67.b.304",
            "esl.67.b.154",
            "esl.67.b.334",
            "esl.67.b.184",
            "esl.67.b.004",
            "esl.67.b.214",
            "esl.67.b.094",
            "esl.67.b.244",
            "esl.67.b.124",
            "esl.67.b.274",
            "esl.67.b.034",
            "esl.67.b.304",
            "esl.67.b.064",
            "esl.67.b.334",
            "esl.67.b.214",
            "bti.66.m.053",
            "bti.66.m.233",
            "bti.66.m.132",
            "bti.66.m.053",
            "bti.66.m.233",
            "bti.66.m.312",
            "bti.66.m.312",
            "bti.66.m.132",
            "mpi.2.a.067",
            "mpi.1.u.157",
            "isl.79.a.",
        ]

        sig_name_with_d = [
            "mpid66m020",
            "mpid66m020",
            "mpid66m067",
            "mpid067u",
            "mpid66m097",
            "mpid097u",
            "mpid66m127",
            "mpid127u",
            "mpid66m157",
            "mpid157u",
            "mpid66m200",
            "mpid66m247",
            "mpid66m277",
            "mpid66m307",
            "mpid66m340",
            "mpid67a022",
            "mpid67a037",
            "mpid67a052",
            "mpid67a067",
            "mpid67a082",
            "mpid67a097",
            "mpid67a142",
            "mpid67a217",
            "mpid67a262",
            "mpid67a277",
            "mpid67a307",
            "mpid67a337",
            "mpid67b022",
            "mpid67b037",
            "mpid67b052",
            "mpid67b097",
            "mpid67b157",
            "mpid67b217",
            "mpid67b277",
            "mpid67b337",
            "mpid79a072",
            "mpid79a147",
            "mpid79a222",
            "mpid79a272",
            "mpid79b067",
            "mpid79b142",
            "mpid79b217",
            "mpid79b277",
            "mpid1a011",
            "mpid1a049",
            "mpid1a109",
            "mpid1a139",
            "mpid1a199",
            "mpid1a244",
            "mpid1a274",
            "mpid1a341",
            "mpid1b011",
            "mpid1b049",
            "mpid1b109",
            "mpid1b139",
            "mpid1b199",
            "mpid1b244",
            "mpid1b274",
            "mpid1b341",
            "mpid5a199",
            "mpid4a199",
            "mpid3a199",
            "mpid2a199",
            "mpid1a199",
            "mpid1b199",
            "mpid2b199",
            "mpid3b199",
            "mpid4b199",
            "mpid5b199",
            "isld66m017",
            "isld66m042",
            "isld66m072",
            "isld079u",
            "isld66m102",
            "isld66m132",
            "isld139u",
            "isld66m197",
            "isld199u",
            "isld66m252",
            "isld66m312",
            "isld67a017",
            "isld67a052",
            "isld67a072",
            "isld67a112",
            "isld67a132",
            "isld67a197",
            "isld67a252",
            "isld67a312",
            "isld67b017",
            "isld67b052",
            "isld67b072",
            "isld67b112",
            "isld67b132",
            "isld67b197",
            "isld67b252",
            "isld67b312",
            "isld79a072",
            "isld79a147",
            "isld79a222",
            "isld79a272",
            "isld79b067",
            "isld79b142",
            "isld79b217",
            "isld79b277",
            "isld1a011",
            "isld1a049",
            "isld1a109",
            "isld1a139",
            "isld1a199",
            "isld1a244",
            "isld1a274",
            "isld1a341",
            "isld1b011",
            "isld1b049",
            "isld1b109",
            "isld1b139",
            "isld1b199",
            "isld1b244",
            "isld1b274",
            "isld1b341",
            "isld5a199",
            "isld4a199",
            "isld3a199",
            "isld2a199",
            "isld1a199",
            "isld1b199",
            "isld2b199",
            "isld3b199",
            "isld4b199",
            "isld5b199",
            "esld66m019",
            "esld66m079",
            "esld079u",
            "esld66m139",
            "esld139u",
            "esld66m199",
            "esld199u",
            "esld66m259",
            "esld66m319",
            "esld079",
            "esld139",
            "esld199",
            "esld67a004",
            "esld67a034",
            "esld67a064",
            "esld67a094",
            "esld67a124",
            "esld67a154",
            "esld67a184",
            "esld67a214",
            "esld67a244",
            "esld67a274",
            "esld67a304",
            "esld67a334",
            "esld67b004",
            "esld67b034",
            "esld67b064",
            "esld67b094",
            "esld67b124",
            "esld67b154",
            "esld67b184",
            "esld67b214",
            "esld67b244",
            "esld67b274",
            "esld67b304",
            "esld67b334",
            "btid66m053",
            "btid66m132",
            "btid66m233",
            "btid66m312",
        ]

        name_with_d = [
            "mpid.66.m.020",
            "mpid.66.m.020",
            "mpid.66.m.067",
            "mpid.067.u.",
            "mpid.66.m.097",
            "mpid.097.u.",
            "mpid.66.m.127",
            "mpid.127.u.",
            "mpid.66.m.157",
            "mpid.157.u.",
            "mpid.66.m.200",
            "mpid.66.m.247",
            "mpid.66.m.277",
            "mpid.66.m.307",
            "mpid.66.m.340",
            "mpid.67.a.022",
            "mpid.67.a.037",
            "mpid.67.a.052",
            "mpid.67.a.067",
            "mpid.67.a.082",
            "mpid.67.a.097",
            "mpid.67.a.142",
            "mpid.67.a.217",
            "mpid.67.a.262",
            "mpid.67.a.277",
            "mpid.67.a.307",
            "mpid.67.a.337",
            "mpid.67.b.022",
            "mpid.67.b.037",
            "mpid.67.b.052",
            "mpid.67.b.097",
            "mpid.67.b.157",
            "mpid.67.b.217",
            "mpid.67.b.277",
            "mpid.67.b.337",
            "mpid.79.a.072",
            "mpid.79.a.147",
            "mpid.79.a.222",
            "mpid.79.a.272",
            "mpid.79.b.067",
            "mpid.79.b.142",
            "mpid.79.b.217",
            "mpid.79.b.277",
            "mpid.1.a.011",
            "mpid.1.a.049",
            "mpid.1.a.109",
            "mpid.1.a.139",
            "mpid.1.a.199",
            "mpid.1.a.244",
            "mpid.1.a.274",
            "mpid.1.a.341",
            "mpid.1.b.011",
            "mpid.1.b.049",
            "mpid.1.b.109",
            "mpid.1.b.139",
            "mpid.1.b.199",
            "mpid.1.b.244",
            "mpid.1.b.274",
            "mpid.1.b.341",
            "mpid.5.a.199",
            "mpid.4.a.199",
            "mpid.3.a.199",
            "mpid.2.a.199",
            "mpid.1.a.199",
            "mpid.1.b.199",
            "mpid.2.b.199",
            "mpid.3.b.199",
            "mpid.4.b.199",
            "mpid.5.b.199",
            "isld.66.m.017",
            "isld.66.m.042",
            "isld.66.m.072",
            "isld.079.u.",
            "isld.66.m.102",
            "isld.66.m.132",
            "isld.139.u.",
            "isld.66.m.197",
            "isld.199.u.",
            "isld.66.m.252",
            "isld.66.m.312",
            "isld.67.a.017",
            "isld.67.a.052",
            "isld.67.a.072",
            "isld.67.a.112",
            "isld.67.a.132",
            "isld.67.a.197",
            "isld.67.a.252",
            "isld.67.a.312",
            "isld.67.b.017",
            "isld.67.b.052",
            "isld.67.b.072",
            "isld.67.b.112",
            "isld.67.b.132",
            "isld.67.b.197",
            "isld.67.b.252",
            "isld.67.b.312",
            "isld.79.a.072",
            "isld.79.a.147",
            "isld.79.a.222",
            "isld.79.a.272",
            "isld.79.b.067",
            "isld.79.b.142",
            "isld.79.b.217",
            "isld.79.b.277",
            "isld.1.a.011",
            "isld.1.a.049",
            "isld.1.a.109",
            "isld.1.a.139",
            "isld.1.a.199",
            "isld.1.a.244",
            "isld.1.a.274",
            "isld.1.a.341",
            "isld.1.b.011",
            "isld.1.b.049",
            "isld.1.b.109",
            "isld.1.b.139",
            "isld.1.b.199",
            "isld.1.b.244",
            "isld.1.b.274",
            "isld.1.b.341",
            "isld.5.a.199",
            "isld.4.a.199",
            "isld.3.a.199",
            "isld.2.a.199",
            "isld.1.a.199",
            "isld.1.b.199",
            "isld.2.b.199",
            "isld.3.b.199",
            "isld.4.b.199",
            "isld.5.b.199",
            "esld.66.m.019",
            "esld.66.m.079",
            "esld.079.u.",
            "esld.66.m.139",
            "esld.139.u.",
            "esld.66.m.199",
            "esld.199.u.",
            "esld.66.m.259",
            "esld.66.m.319",
            "esld.079..",
            "esld.139..",
            "esld.199..",
            "esld.67.a.004",
            "esld.67.a.034",
            "esld.67.a.064",
            "esld.67.a.094",
            "esld.67.a.124",
            "esld.67.a.154",
            "esld.67.a.184",
            "esld.67.a.214",
            "esld.67.a.244",
            "esld.67.a.274",
            "esld.67.a.304",
            "esld.67.a.334",
            "esld.67.b.004",
            "esld.67.b.034",
            "esld.67.b.064",
            "esld.67.b.094",
            "esld.67.b.124",
            "esld.67.b.154",
            "esld.67.b.184",
            "esld.67.b.214",
            "esld.67.b.244",
            "esld.67.b.274",
            "esld.67.b.304",
            "esld.67.b.334",
            "btid.66.m.053",
            "btid.66.m.132",
            "btid.66.m.233",
            "btid.66.m.312",
        ]
        sig_names = sig_name_with_d + sig_name_without_d
        names = name_with_d + name_without_d

        for name in sig_names:
            signals.append(PtDataSignal(name))

    elif diag_name == "custom":
        sig_names = sig_names_custom
        names = names_custom
        for name in sig_names:
            if tree_custom == "ptdata":
                signals.append(PtDataSignal(name))
            else:
                signals.append(
                    MdsSignal(name, tree_custom, location="remote://atlas.gat.com")
                )

    elif diag_name == "mag_hi":
        sig_names = [f"b{i}" for i in range(1, 9)]
        names = [f"b{i}" for i in range(1, 9)]
        for name in sig_names:
            signals.append(PtDataSignal(name))

    elif diag_name == "bes":
        sig_names = [f"besfu{i:02}" for i in range(1, 65)]

        sig_names.append("bes_r")
        sig_names.append("bes_z")
        names = [f"{i:02}" for i in range(1, 65)]

        names.append("r")
        names.append("z")

        for name in sig_names:
            signals.append(PtDataSignal(name))

    elif diag_name == "ece_cali":
        channels = range(1, 49)  # 48

        for chan in channels:
            name = rf"\TECEF{chan:02d}"
            signals.append(MdsSignal(name, "ECE", location="remote://atlas.gat.com"))
            names.append(f"{chan:02d}")

    elif diag_name == "ece_s":
        channels = range(1, 49)  # 48

        for chan in channels:
            name = rf"\TECE{chan:02d}"
            signals.append(MdsSignal(name, "ECE", location="remote://atlas.gat.com"))
            names.append(f"{chan:02d}")

    elif diag_name == "co2_s":
        chords = ["r0", "v1", "v2", "v3"]

        for chord in chords:
            name = rf"\den{chord}"
            signals.append(MdsSignal(name, "BCI", location="remote://atlas.gat.com"))
            names.append(f"{chord}")

    elif diag_name == "co2_den":
        nums = range(1, 15)
        chords = ["r0", "v1", "v2", "v3"]
        phases = ["den"]

        for _phase in phases:
            for chord in chords:
                for num in nums:
                    name = rf"\den{chord}_uf_{num}"
                    signals.append(
                        MdsSignal(name, "BCI", location="remote://atlas.gat.com")
                    )
                    names.append(f"{chord}_{num}")

    elif diag_name == "co2_pl":
        nums = range(1, 15)
        chords = ["r0", "v1", "v2", "v3"]
        phases = ["pl"]

        for _phase in phases:
            for chord in chords:
                for num in nums:
                    name = rf"\pl1{chord}_uf_{num}"
                    signals.append(
                        MdsSignal(name, "BCI", location="remote://atlas.gat.com")
                    )
                    names.append(f"{chord}_{num}")

    elif diag_name == "ts":
        # thomson_mds_scale={'density': 1e19, 'temp': 1e3}
        thomson_mds_areas = ["core", "divertor", "tangential"]
        thomson_sig_names = ["density", "temp"]
        thomson_names = ["dens", "temp"]
        treename = "electrons"

        for thomson_mds_area in thomson_mds_areas:
            for thomson_sig_name, thomson_name in zip(thomson_sig_names, thomson_names):
                name = rf"TS.BLESSED.{thomson_mds_area}.{thomson_sig_name}"

                signals.append(
                    MdsSignal(name, treename, location="remote://atlas.gat.com")
                )

                names.append(rf"{thomson_mds_area}.{thomson_name}")

    elif diag_name == "ts_rz":
        # thomson_mds_scale={'density': 1e19, 'temp': 1e3}
        thomson_mds_areas = ["core", "divertor", "tangential"]
        thomson_sig_names = ["r", "z"]
        treename = "electrons"

        for thomson_mds_area in thomson_mds_areas:
            for thomson_sig_name in thomson_sig_names:
                name = rf"TS.BLESSED.{thomson_mds_area}.{thomson_sig_name}"
                signals.append(
                    MdsSignal(name, treename, location="remote://atlas.gat.com")
                )

                names.append(rf"{thomson_mds_area}.{thomson_sig_name}")

    elif diag_name == "TS_ERROR":
        # thomson_mds_scale={'density': 1e19, 'temp': 1e3}
        thomson_mds_areas = ["core", "divertor", "tangential"]
        thomson_sig_names = ["DENSITY_E", "TEMP_E"]
        thomson_names = ["dens", "temp"]
        treename = "electrons"

        for thomson_mds_area in thomson_mds_areas:
            for thomson_sig_name, thomson_name in zip(thomson_sig_names, thomson_names):
                name = rf"\TS_{thomson_mds_area}_{thomson_sig_name}"
                signals.append(
                    MdsSignal(name, treename, location="remote://atlas.gat.com")
                )

                names.append(rf"{thomson_mds_area}.{thomson_name}")

    elif diag_name == "mag":
        sig_names = [
            "DSL1U180",
            "DSL2U180",
            "DSL3U180",
            "DSL4U157",
            "DSL5U157",
            "DSL6U157",
            "MPI11M067",
            "MPI11M322",
            "MPI1A139",
            "MPI1A322",
            "MPI1B139",
            "MPI1B157",
            "MPI1B322",
            "MPI1L180",
            "MPI1U157",
            "MPI2A067",
            "MPI2A139",
            "MPI2A322",
            "MPI2B067",
            "MPI2B139",
            "MPI2B322",
            "MPI2L180",
            "MPI2U157",
            "MPI3A139",
            "MPI3A322",
            "MPI3B139",
            "MPI3B322",
            "MPI3L180",
            "MPI3U157",
            "MPI4A139",
            "MPI4A322",
            "MPI4B139",
            "MPI4B322",
            "MPI4U157",
            "MPI5A139",
            "MPI5A322",
            "MPI5B139",
            "MPI5B322",
            "MPI5U157",
            "MPI66M067",
            "MPI66M157",
            "MPI66M247",
            "MPI66M322",
            "MPI67A097",
            "MPI67A142",
            "MPI67A157",
            "MPI67A322",
            "MPI67B097",
            "MPI67B157",
            "MPI67B322",
            "MPI6FA322",
            "MPI6FB142",
            "MPI6FB322",
            "MPI6NA132",
            "MPI6NA157",
            "MPI6NA322",
            "MPI6NB157",
            "MPI6NB322",
            "MPI6U157",
            "MPI79A147",
            "MPI79B142",
            "MPI79B322",
            "MPI79FA322",
            "MPI79NA322",
            "MPI7FA322",
            "MPI7FB322",
            "MPI7NA322",
            "MPI7NB142",
            "MPI7NB322",
            "MPI7U157",
            "MPI89A322",
            "MPI89B322",
            "MPI8A322",
            "MPI8B322",
            "MPI9A322",
            "MPI9B322",
            "PSF1A",
            "PSF1B",
            "PSF2A",
            "PSF2B",
            "PSF3A",
            "PSF3B",
            "PSF4A",
            "PSF4B",
            "PSF5A",
            "PSF5B",
            "PSF6FA",
            "PSF6FB",
            "PSF6NA",
            "PSF6NB",
            "PSF7FA",
            "PSF7FB",
            "PSF7NA",
            "PSF7NB",
            "PSF8A",
            "PSF8B",
            "PSF9A",
            "PSF9B",
            "PSI11M",
            "PSI12A",
            "PSI12B",
            "PSI1L",
            "PSI23A",
            "PSI23B",
            "PSI2L",
            "PSI34A",
            "PSI34B",
            "PSI3L",
            "PSI45A",
            "PSI45B",
            "PSI58A",
            "PSI58B",
            "PSI6A",
            "PSI6B",
            "PSI7A",
            "PSI7B",
            "PSI89FB",
            "PSI89NB",
            "PSI9A",
            "PSI9B",
        ]
        # Poloidal Flux Loops (Wb/rad):  psf, psi.
        # Poloidal Field Probes: mpi, dsl

        # E-coil Currents (A): can be found in actu
        # F-coil Currents (A): can be found in actu
        # I-coil Currents (A): can be found in actu
        # C-coil Currents (A): can be found in actu

        # Miscellaneous data: can be found in basic

        # https://diii-d.gat.com/d3d-wiki/images/6/68/Mag_eq_2013_LABEL.pdf
        names = [
            "dsl.1u180",
            "dsl.2u180",
            "dsl.3u180",
            "dsl.4u157",
            "dsl.5u157",
            "dsl.6u157",
            "mpi.11m067",
            "mpi.11m322",
            "mpi.1a139",
            "mpi.1a322",
            "mpi.1b139",
            "mpi.1b157",
            "mpi.1b322",
            "mpi.1l180",
            "mpi.1u157",
            "mpi.2a067",
            "mpi.2a139",
            "mpi.2a322",
            "mpi.2b067",
            "mpi.2b139",
            "mpi.2b322",
            "mpi.2l180",
            "mpi.2u157",
            "mpi.3a139",
            "mpi.3a322",
            "mpi.3b139",
            "mpi.3b322",
            "mpi.3l180",
            "mpi.3u157",
            "mpi.4a139",
            "mpi.4a322",
            "mpi.4b139",
            "mpi.4b322",
            "mpi.4u157",
            "mpi.5a139",
            "mpi.5a322",
            "mpi.5b139",
            "mpi.5b322",
            "mpi.5u157",
            "mpi.66m067",
            "mpi.66m157",
            "mpi.66m247",
            "mpi.66m322",
            "mpi.67a097",
            "mpi.67a142",
            "mpi.67a157",
            "mpi.67a322",
            "mpi.67b097",
            "mpi.67b157",
            "mpi.67b322",
            "mpi.6fa322",
            "mpi.6fb142",
            "mpi.6fb322",
            "mpi.6na132",
            "mpi.6na157",
            "mpi.6na322",
            "mpi.6nb157",
            "mpi.6nb322",
            "mpi.6u157",
            "mpi.79a147",
            "mpi.79b142",
            "mpi.79b322",
            "mpi.79fa322",
            "mpi.79na322",
            "mpi.7fa322",
            "mpi.7fb322",
            "mpi.7na322",
            "mpi.7nb142",
            "mpi.7nb322",
            "mpi.7u157",
            "mpi.89a322",
            "mpi.89b322",
            "mpi.8a322",
            "mpi.8b322",
            "mpi.9a322",
            "mpi.9b322",
            "psf.1a",
            "psf.1b",
            "psf.2a",
            "psf.2b",
            "psf.3a",
            "psf.3b",
            "psf.4a",
            "psf.4b",
            "psf.5a",
            "psf.5b",
            "psf.6fa",
            "psf.6fb",
            "psf.6na",
            "psf.6nb",
            "psf.7fa",
            "psf.7fb",
            "psf.7na",
            "psf.7nb",
            "psf.8a",
            "psf.8b",
            "psf.9a",
            "psf.9b",
            "psi.11m",
            "psi.12a",
            "psi.12b",
            "psi.1l",
            "psi.23a",
            "psi.23b",
            "psi.2l",
            "psi.34a",
            "psi.34b",
            "psi.3l",
            "psi.45a",
            "psi.45b",
            "psi.58a",
            "psi.58b",
            "psi.6a",
            "psi.6b",
            "psi.7a",
            "psi.7b",
            "psi.89fb",
            "psi.89nb",
            "psi.9a",
            "psi.9b",
        ]

        for name in sig_names:
            signals.append(PtDataSignal(name))

    elif diag_name == "cer":
        channels = [f"v{i}" for i in range(1, 33)] + [f"t{i}" for i in range(1, 49)]

        name_channels = [f"v{i:02d}" for i in range(1, 33)] + [
            f"t{i:02d}" for i in range(1, 49)
        ]
        outputs = [
            "amp",
            "samp",
            "ti",
            "sti",
            "rot",
            "srot",
            "r",
            "phi",
            "nz",
            "fz",
            "zeff",
            "vb",
            "svb",
        ]

        sig_names = [
            rf"\cerq{output}{channel}" for channel in channels for output in outputs
        ]
        names = [
            rf"cer.{output}.{channel}"
            for channel in name_channels
            for output in outputs
        ]

        treename = "ions"
        signals = []
        for name in sig_names:
            signals.append(MdsSignal(name, treename, location="remote://atlas.gat.com"))

    elif diag_name == "mse":
        treename = "mse"
        sig_names = [rf"\msep{i:02d}" for i in range(1, 70)]
        names = [f"{i:02d}" for i in range(1, 70)]

        signals = []
        for name in sig_names:
            signals.append(MdsSignal(name, treename, location="remote://atlas.gat.com"))

    return names, signals


def fetch_ece_2d_array_data(
    path: str,
    shots: list[int],
    diag_name: str,
) -> None:
    names, signals = signal_gen(diag_name)

    for n, shot in tqdm(enumerate(shots)):
        if shot >= start_shot:
            time.time()
            pipeline = Pipeline([shot])

            for i, name in enumerate(names):
                pipeline.fetch(name, signals[i])

            records = pipeline.compute_serial()
            shot_data = dict(records[0])
            data_h5 = {}

            try:
                data_h5["xdata"] = shot_data[names[0]]["times"]
                data_h5["xunits"] = shot_data[names[0]]["units"]["times"]
                data_h5["yunits"] = ""
                data_h5["zunits"] = shot_data[names[0]]["units"]["data"]
                data_tmp = []
                for name in names:
                    try:
                        len_tmp = len(shot_data[name]["data"])
                        if len_tmp < 10:
                            break
                    except (KeyError, IndexError):
                        break
                    # print(len(shot_data[name]['data']))
                    data_tmp.append(shot_data[name]["data"][: len(data_h5["xdata"])])

                data_tmp = np.array(data_tmp, dtype="float")

                data_h5["zdata"] = data_tmp
                data_h5["ydata"] = []

                if len(data_h5["xdata"]) > 3500000:
                    data_h5["xdata"] = data_h5["xdata"][:3500000]
                    data_h5["zdata"] = data_h5["zdata"][:, :3500000]
            except (KeyError, IndexError, ValueError):
                data_h5["xdata"] = []
                data_h5["ydata"] = []
                data_h5["zdata"] = []

                data_h5["xunits"] = ""
                data_h5["yunits"] = ""
                data_h5["zunits"] = ""

            with h5py.File(f"{path}{shot}_{diag_name}.h5", "w") as h5file:
                save_dict_to_hdf5(data_h5, h5file)

        if n % interval == 0:
            size_limiter_sleep(size_GB=size_GB)
            print(f"shot={shot}")
            print(n, flush=True)


# fetching the data
def fetch_single_data(
    path: str,
    shots: list[int],
    diag_name: str,
) -> None:
    names, signals = signal_gen(diag_name)

    if not os.path.isdir(path):
        os.makedirs(path)

    for n, shot in tqdm(enumerate(shots)):
        if shot >= start_shot:
            if 1 == 1:
                time.time()
                pipeline = Pipeline([shot])

                for i, name in enumerate(names):
                    pipeline.fetch(name, signals[i])

                records = pipeline.compute_serial()
                shot_data = dict(records[0])
                # print(shot_data.keys())
                ##print(shot_data)
                data_h5 = {}
                # print(names)
                for name in names:
                    data_h5[name] = {}
                    # print(data_h5)
                    data_tmp = shot_data[name]
                    # print(name)

                    try:
                        total_len = len(data_tmp["times"])
                        dt = data_tmp["times"][1] - data_tmp["times"][0]
                        cut_index = int(np.min([total_len + 1, 7500 / dt]))

                        if diag_name in diag_3d:
                            # print(data_tmp.keys())
                            data_h5[name]["xdata"] = data_tmp["times"][:cut_index]
                            data_h5[name]["ydata"] = data_tmp["rhon"][:]
                            data_h5[name]["zdata"] = data_tmp["data"][:cut_index]

                            data_h5[name]["xunits"] = data_tmp["units"]["times"]
                            data_h5[name]["yunits"] = "rhon"
                            data_h5[name]["zunits"] = data_tmp["units"]["data"]
                        else:
                            # print(data_tmp.keys())
                            data_h5[name]["xdata"] = data_tmp["times"][:cut_index]
                            data_h5[name]["ydata"] = []
                            data_h5[name]["zdata"] = data_tmp["data"][:cut_index]

                            data_h5[name]["xunits"] = data_tmp["units"]["times"]
                            data_h5[name]["yunits"] = ""
                            data_h5[name]["zunits"] = data_tmp["units"]["data"]
                    except (KeyError, IndexError, ValueError):
                        data_h5[name]["xdata"] = []
                        data_h5[name]["ydata"] = []
                        data_h5[name]["zdata"] = []

                        data_h5[name]["xunits"] = ""
                        data_h5[name]["yunits"] = ""
                        data_h5[name]["zunits"] = ""

                # print(data_h5.keys())
                # Save to h5
                with h5py.File(f"{path}{shot}_{diag_name}.h5", "w") as hf:
                    for key in data_h5.keys():
                        group = hf.create_group(key)
                        for subkey, value in data_h5[key].items():
                            # Check if the value is a string type
                            if isinstance(value, np.ndarray) and np.issubdtype(
                                value.dtype, np.str_
                            ):
                                # Special dtype for string data
                                str_dtype = h5py.string_dtype(encoding="utf-8")
                                group.create_dataset(
                                    subkey,
                                    data=value.astype(object),
                                    dtype=str_dtype,
                                )
                            else:
                                group.create_dataset(subkey, data=value)

            if 1 == 2:
                pass
            if n % interval == 0:
                size_limiter_sleep(size_GB=size_GB)
                print(f"shot={shot}")
                print(n, flush=True)


def fetch_co2_chunked_data(
    path: str,
    shots: list[int],
    diag_name: str,
) -> None:
    chords = ["r0", "v1", "v2", "v3"]
    nums = range(1, 15)
    names, signals = signal_gen(diag_name)
    for n, shot in tqdm(enumerate(shots)):
        if shot >= start_shot:
            try:
                time.time()
                pipeline = Pipeline([shot])

                for i, name in enumerate(names):
                    pipeline.fetch(name, signals[i])

                records = pipeline.compute_serial()
                shot_data = dict(records[0])
                data_h5 = {}
                for chord in chords:
                    data_h5[chord] = {}
                for num in nums:
                    for _i_chord, chord in enumerate(chords):
                        # print(f'num={num}')
                        # print(f'chord={chord}')

                        data_tmp = shot_data[f"{chord}_{num}"]

                        try:
                            len(data_tmp["data"])
                            # print(len(data_tmp['data']))
                        except (KeyError, TypeError, AttributeError):
                            break

                        if num == 1:
                            data_h5[chord]["xdata"] = data_tmp["times"]
                            data_h5[chord]["ydata"] = []
                            data_h5[chord]["zdata"] = data_tmp["data"]

                            data_h5[chord]["xunits"] = data_tmp["units"]["times"]
                            data_h5[chord]["yunits"] = ""
                            data_h5[chord]["zunits"] = data_tmp["units"]["data"]
                        else:
                            data_h5[chord]["xdata"] = np.concatenate(
                                [data_h5[chord]["xdata"], data_tmp["times"]]
                            )
                            data_h5[chord]["zdata"] = np.concatenate(
                                [data_h5[chord]["zdata"], data_tmp["data"]]
                            )
                shot_data = None

                # Save to h5
                with h5py.File(f"{path}{shot}_{diag_name}.h5", "w") as hf:
                    for key in data_h5.keys():
                        group = hf.create_group(key)
                        for subkey, value in data_h5[key].items():
                            # Check if the value is a string type
                            if isinstance(value, np.ndarray) and np.issubdtype(
                                value.dtype, np.str_
                            ):
                                # Special dtype for string data
                                str_dtype = h5py.string_dtype(encoding="utf-8")
                                group.create_dataset(
                                    subkey,
                                    data=value.astype(object),
                                    dtype=str_dtype,
                                )
                            else:
                                group.create_dataset(subkey, data=value)
            except Exception:
                pass

            if n % interval == 0:
                size_limiter_sleep(size_GB=size_GB)
                print(f"shot={shot}")
                print(n, flush=True)


def fetch_co2_chunked_data_2d(
    path: str,
    shots: list[int],
    diag_name: str,
) -> None:
    chords = ["r0", "v1", "v2", "v3"]
    nums = range(1, 15)
    names, signals = signal_gen(diag_name)
    for n, shot in enumerate(shots):
        if shot >= start_shot:
            try:
                time.time()
                pipeline = Pipeline([shot])

                for i, name in enumerate(names):
                    pipeline.fetch(name, signals[i])

                records = pipeline.compute_serial()
                shot_data = dict(records[0])
                data_h5 = {}
                for chord in chords:
                    data_h5[chord] = {}
                for num in nums:
                    for _i_chord, chord in enumerate(chords):
                        # print(f'num={num}')
                        # print(f'chord={chord}')

                        data_tmp = shot_data[f"{chord}_{num}"]

                        try:
                            len(data_tmp["data"])
                            # print(len(data_tmp['data']))
                        except (KeyError, TypeError, AttributeError):
                            break

                        if num == 1:
                            data_h5[chord]["xdata"] = data_tmp["times"]
                            data_h5[chord]["ydata"] = []
                            data_h5[chord]["zdata"] = data_tmp["data"]

                            data_h5[chord]["xunits"] = data_tmp["units"]["times"]
                            data_h5[chord]["yunits"] = ""
                            data_h5[chord]["zunits"] = data_tmp["units"]["data"]
                        else:
                            data_h5[chord]["xdata"] = np.concatenate(
                                [data_h5[chord]["xdata"], data_tmp["times"]]
                            )
                            data_h5[chord]["zdata"] = np.concatenate(
                                [data_h5[chord]["zdata"], data_tmp["data"]]
                            )
                shot_data = None
                data_h5_new = {}
                data_h5_new["xdata"] = np.array(data_h5[chords[0]]["xdata"])
                data_h5_new["zdata"] = np.array(
                    [data_h5[chord]["zdata"] for chord in chords]
                )
                data_h5_new["keys"] = np.array(chords)

                data_h5 = None
                # Save to h5
                with h5py.File(f"{path}{shot}_{diag_name}.h5", "w") as hf:
                    for key in data_h5_new.keys():
                        group = hf.create_group(key)
                        for subkey, value in data_h5_new[key].items():
                            # Check if the value is a string type
                            if isinstance(value, np.ndarray) and np.issubdtype(
                                value.dtype, np.str_
                            ):
                                # Special dtype for string data
                                str_dtype = h5py.string_dtype(encoding="utf-8")
                                group.create_dataset(
                                    subkey,
                                    data=value.astype(object),
                                    dtype=str_dtype,
                                )
                            else:
                                group.create_dataset(subkey, data=value)
            except Exception:
                pass

            if n % interval == 0:
                size_limiter_sleep(size_GB=size_GB)
                print(f"shot={shot}")
                print(n, flush=True)


def fetch_data(path: str, shots: list[int], diag_name: str) -> None:
    if diag_name in ["co2_den", "co2_pl"]:
        fetch_co2_chunked_data(path, shots, diag_name)
    elif diag_name in ["ece_s", "ece_cali"]:
        fetch_ece_2d_array_data(path, shots, diag_name)
    else:
        fetch_single_data(path, shots, diag_name)


if __name__ == "__main__":
    fetch_data(path, shots, diag_name)
