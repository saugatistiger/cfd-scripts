"""
Convert a LIGGGHTS custom dump snapshot (id type x y z vx vy vz fx fy fz
omegax omegay omegaz diameter c_cont) into a LAMMPS/LIGGGHTS data file
(atom_style sphere) that can be used as an initial configuration with
`read_data`.

Usage:
    python3 dump_to_data.py
"""

DUMP_FILE = "/work/e773/e773/ss1604/20260820/40_square/SIM009_us8.896/CFD_new1/post/dump_atom_940000.packing"
OUTPUT_FILE="/work/e773/e773/ss1604/20260820/40_square/SIM009_us8.896/DEM_BED/output_settled.data"

# Density is not stored in the dump file (only diameter is). Reuse the
# density used in the earlier settled data file (2.65) since it's the
# same particle set/material.
DENSITY = 2.6499999999999999e+00


def read_dump(path):
    with open(path) as f:
        lines = f.readlines()

    idx = 0
    assert lines[idx].startswith("ITEM: TIMESTEP")
    timestep = int(lines[idx + 1])
    idx += 2

    assert lines[idx].startswith("ITEM: NUMBER OF ATOMS")
    natoms = int(lines[idx + 1])
    idx += 2

    assert lines[idx].startswith("ITEM: BOX BOUNDS")
    xlo, xhi = map(float, lines[idx + 1].split())
    ylo, yhi = map(float, lines[idx + 2].split())
    zlo, zhi = map(float, lines[idx + 3].split())
    idx += 4

    header_line = lines[idx]
    assert header_line.startswith("ITEM: ATOMS")
    cols = header_line.replace("ITEM: ATOMS", "").split()
    idx += 1

    col_idx = {name: i for i, name in enumerate(cols)}

    atoms = []
    for i in range(natoms):
        parts = lines[idx + i].split()
        atoms.append(parts)

    return {
        "timestep": timestep,
        "natoms": natoms,
        "box": (xlo, xhi, ylo, yhi, zlo, zhi),
        "col_idx": col_idx,
        "atoms": atoms,
    }


def write_data(path, dump):
    col = dump["col_idx"]
    xlo, xhi, ylo, yhi, zlo, zhi = dump["box"]

    with open(path, "w") as f:
        f.write(
            "LAMMPS data file via write_data, version Version "
            "LIGGGHTS-PUBLIC 3.8.0, timestep = {}\n\n".format(dump["timestep"])
        )
        f.write("{} atoms\n".format(dump["natoms"]))
        f.write("1 atom types\n\n")
        f.write("{:.16e} {:.16e} xlo xhi\n".format(xlo, xhi))
        f.write("{:.16e} {:.16e} ylo yhi\n".format(ylo, yhi))
        f.write("{:.16e} {:.16e} zlo zhi\n\n".format(zlo, zhi))

        f.write("Atoms\n\n")
        for parts in dump["atoms"]:
            aid = parts[col["id"]]
            atype = parts[col["type"]]
            diameter = parts[col["diameter"]]
            x = parts[col["x"]]
            y = parts[col["y"]]
            z = parts[col["z"]]
            f.write(
                "{} {} {} {:.16e} {} {} {} 0 0 0\n".format(
                    aid, atype, diameter, DENSITY, x, y, z
                )
            )

        f.write("\nVelocities\n\n")
        for parts in dump["atoms"]:
            aid = parts[col["id"]]
            vx = parts[col["vx"]]
            vy = parts[col["vy"]]
            vz = parts[col["vz"]]
            omegax = parts[col["omegax"]]
            omegay = parts[col["omegay"]]
            omegaz = parts[col["omegaz"]]
            f.write(
                "{} {} {} {} {} {} {}\n".format(
                    aid, vx, vy, vz, omegax, omegay, omegaz
                )
            )


def main():
    dump = read_dump(DUMP_FILE)
    write_data(OUTPUT_FILE, dump)
    print(
        "Converted {} atoms from timestep {} -> {}".format(
            dump["natoms"], dump["timestep"], OUTPUT_FILE
        )
    )


if __name__ == "__main__":
    main()
