"""
Packs the runs of a sweep into json files the viewer can be given.

``viewer.py`` writes a page with the runs of one rung baked into it, which is what a
single case wants. A sweep is the other shape: five rungs, several horizons and five
ways of posing the problem, all solved on a machine whose result folders are hundreds
of megabytes of h5 that no browser opens. The whole sweep baked into one page would be
tens of megabytes, and most of it would never be looked at.

So the page is written once, empty, and every run is packed on its own. A packed run is
what the map needs and nothing else, a few hundred kilobytes, and it carries the job it
came from, i.e. which rung, which horizon, what was decided and how the solve went. The
page is then opened from disk and the runs worth seeing are dropped onto it.

Three things are joined into one packed run, which is why the h5 on its own is not
enough to draw anything:

- ``optimization_results.h5`` of the result folder, i.e. what was built and how it is
  operated. It names the nodes but does not say where they are
- ``NodeLocations.csv`` of the rung, i.e. the coordinates the map is drawn on
- the row of ``results.csv``, i.e. which rung the run belongs to, what the job decided
  and how the solve went

The locations come from the rung next to this script rather than from the copy of the
case the job worked on, because ``sweep.py`` deletes that copy unless it is told to
keep it. The rungs are fixed, so the locations are the same either way.

A folder that is not a sweep is packed the same way, which is what a single run of
``run.py`` needs: one result folder, or the ``userData`` of a rung holding many. Such a
folder carries no ``results.csv``, so nothing in it says which rung it belongs to and
``--rung`` has to say it instead.

Run with::

    python sweep_pack.py <sweep folder>              every run of the sweep
    python sweep_pack.py <sweep folder> --rung 3L3S  one rung of it
    python sweep_pack.py <sweep folder> --hours 336  one horizon of it
    python sweep_pack.py <sweep folder> --bundle     all of it in one file
    python sweep_pack.py 2L2S/userData --rung 2L2S   every run of one rung
    python sweep_pack.py 2L2S/userData/20260917T1830 --rung 2L2S    one of them
    python sweep_pack.py --viewer                    only the empty page

The files land in a ``packed/`` folder, next to an ``index.csv`` saying what each one
holds, and the empty page is written there as ``viewer.html``.

.. note::
    A job whose solver found no solution still has an h5, but it holds no design, so
    such runs are reported as skipped rather than packed.
"""

import argparse
import csv
import json
from pathlib import Path

import viewer

LADDER_DIR = Path(__file__).parent

#: What the result folders of a job are called, in the order they are solved. The same
#: order ``sweep.py`` reads its logs in, see ``sweep.MODELS``.
MODELS = ["reference", "linepack"]

#: Columns of ``results.csv`` a packed run carries, i.e. what the job was and how the
#: solve went. The rest of the table is about the search rather than about the result.
JOB_COLUMNS = [
    "id",
    "axis",
    "rung",
    "cycles",
    "hours",
    "design",
    "types",
    "corridors",
    "status",
    "gap",
    "first_s",
    "solve_s",
    "rows",
    "cols",
    "binaries",
]


def read_jobs(sweep_dir: Path) -> dict:
    """
    What ``results.csv`` says about every job and network type of the sweep.

    :param Path sweep_dir: folder of the sweep
    :return: the row of each job and model, keyed by both
    """
    table = sweep_dir / "results.csv"
    if not table.is_file():
        raise FileNotFoundError(f"No results.csv under {sweep_dir}")
    with open(table, newline="", encoding="utf-8") as handle:
        return {
            (row["id"], row["model"]): {
                key: row[key] for key in JOB_COLUMNS if row.get(key) != ""
            }
            for row in csv.DictReader(handle)
        }


def result_folders(job_dir: Path) -> list:
    """
    The result folders of one job, in the order the models were solved.

    The folders of ADOPT are named after the second the run started in, so sorting them
    puts the reference first and the linepack second.

    :param Path job_dir: folder of the job inside the sweep
    :return: the folders holding optimization_results.h5
    """
    results = job_dir / "results"
    if not results.is_dir():
        return []
    return sorted(
        path
        for path in results.iterdir()
        if (path / "optimization_results.h5").exists()
    )


def pack(job_dir: Path, jobs: dict) -> list:
    """
    The runs of one job, ready to be written.

    :param Path job_dir: folder of the job inside the sweep
    :param dict jobs: what ``results.csv`` says, see :func:`read_jobs`
    :return: one entry per model solved, each the data of the page plus its job
    """
    packed = []
    for model, folder in zip(MODELS, result_folders(job_dir)):
        job = jobs.get((job_dir.name, model))
        if job is None:
            continue
        run = viewer.read_results(folder, LADDER_DIR / job["rung"])
        if not run["arcs"]:
            continue
        # which model a folder is comes from the order it was solved in rather than
        # from the h5: a linepack run that built nothing writes no linepack either and
        # would otherwise be labelled a reference
        run["kind"] = model
        run["job"] = job
        packed.append(run)
    return packed


def loose_folders(path: Path) -> list:
    """
    The result folders of a path that is not a sweep.

    Either the path is a result folder itself, i.e. what ``run.py`` wrote, or it holds
    them, i.e. the ``userData`` of a rung.

    :param Path path: a result folder, or a folder of them
    :return: the folders holding optimization_results.h5, oldest first
    """
    if (path / "optimization_results.h5").exists():
        return [path]
    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "optimization_results.h5").exists()
    )


def pack_loose(folders: list, rung: str) -> list:
    """
    The runs of a folder that is not a sweep, ready to be written.

    Nothing outside a sweep says which rung a result folder belongs to, so the rung is
    given rather than read, and it is written into the run so that the page can put it
    in its case menu. Which network type the run is comes from the h5 itself: a run of
    the reference network writes no linepack.

    :param list folders: result folders, see :func:`loose_folders`
    :param str rung: rung the folders were solved on
    :return: one entry per folder that holds a design
    """
    packed = []
    for folder in folders:
        run = viewer.read_results(folder, LADDER_DIR / rung)
        if not run["arcs"]:
            print(f"skipped {folder.name}: no design in its results")
            continue
        run["rung"] = rung
        packed.append(run)
    return packed


def write_viewer(output: Path):
    """
    Writes the page with no run in it, ready to be dropped on.

    :param Path output: file to write
    """
    viewer.write_html([], output)


def parse_args(argv=None):
    """
    Reads which sweep to pack and how much of it.

    :param list argv: arguments to read, the command line when left out
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        help="folder of a sweep, i.e. the one holding results.csv, or a result folder "
        "of a rung, or the userData holding them",
    )
    parser.add_argument(
        "--rung",
        default="",
        help="rungs to pack, comma separated. Outside a sweep nothing says which rung "
        "a result folder is, so exactly one has to be named",
    )
    parser.add_argument("--hours", default="", help="horizons to pack, comma separated")
    parser.add_argument("--design", default="", help="what is decided, comma separated")
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="write every run into one file rather than one file each, which is only "
        "worth it for a few runs: the page loads the whole file at once",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="write only the empty page, next to this script",
    )
    parser.add_argument("-o", "--out", default=None, help="where the files are written")
    return parser.parse_args(argv)


def write_all(entries: list, out_dir: Path, bundle: bool) -> list:
    """
    Writes the packed runs, the table saying what they are and the empty page.

    :param list entries: the runs, each with the name its file takes
    :param Path out_dir: folder the files are written to
    :param bool bundle: whether every run goes into one file
    :return: one index row per run
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, run in entries:
        size = 0
        if not bundle:
            path = out_dir / name
            path.write_text(json.dumps(run, separators=(",", ":")), encoding="utf-8")
            size = path.stat().st_size
        index.append(
            {
                **run.get("job", {"rung": run.get("rung", ""), "id": run["run"]}),
                "model": run["kind"],
                "hours": run["horizon"],
                "file": "bundle.json" if bundle else name,
                "kb": round(size / 1024),
            }
        )

    if bundle:
        path = out_dir / "bundle.json"
        runs = [run for _, run in entries]
        path.write_text(json.dumps(runs, separators=(",", ":")), encoding="utf-8")
        for entry in index:
            entry["kb"] = round(path.stat().st_size / 1024 / len(index))

    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=JOB_COLUMNS + ["model", "file", "kb"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for entry in index:
            writer.writerow(entry)

    write_viewer(out_dir / "viewer.html")
    return index


def sweep_entries(sweep_dir: Path, wanted: dict) -> list:
    """
    Every run of a sweep that passes the filters, with the name its file takes.

    :param Path sweep_dir: folder of the sweep
    :param dict wanted: values kept per column, an empty set keeping everything
    :return: the runs, each with its file name
    """
    jobs = read_jobs(sweep_dir)
    entries = []
    for job_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        job = jobs.get((job_dir.name, MODELS[0]))
        if job is None:
            continue
        if any(values and job[key] not in values for key, values in wanted.items()):
            continue
        try:
            packed = pack(job_dir, jobs)
        except (KeyError, OSError, ValueError) as error:
            print(f"skipped {job_dir.name}: {type(error).__name__}: {error}")
            continue
        if not packed:
            print(f"skipped {job_dir.name}: no design in its results")
            continue
        for run in packed:
            entries.append((f"{job_dir.name}.{run['kind']}.json", run))
        print(f"packed  {job_dir.name:34}{len(packed)} run(s)")
    return entries


def loose_entries(path: Path, rung: str) -> list:
    """
    Every run of a folder that is not a sweep, with the name its file takes.

    :param Path path: a result folder, or a folder of them
    :param str rung: rung the folders were solved on
    :return: the runs, each with its file name
    """
    entries = []
    for run in pack_loose(loose_folders(path), rung):
        kind = run["kind"].replace(" ", "_")
        entries.append((f"{rung}.{run['run']}.{kind}.json", run))
        print(f"packed  {run['run']:24}{run['kind']:12}{run['horizon']:>5} h")
    return entries


def main(argv=None):
    """
    Packs whatever the path holds and writes the page next to it.

    :param list argv: arguments to read, the command line when left out
    """
    args = parse_args(argv)
    if args.viewer and not args.path:
        output = Path(args.out) if args.out else LADDER_DIR / "viewer.html"
        write_viewer(output)
        print(f"written   : {output}")
        return
    if not args.path:
        raise SystemExit("Give a folder to pack, or --viewer for the empty page")

    path = Path(args.path)
    rungs = [value.strip() for value in args.rung.split(",") if value.strip()]

    # a sweep is told apart by its table, which is also the only thing that says which
    # rung each of its jobs was solved on
    if (path / "results.csv").is_file():
        entries = sweep_entries(
            path,
            {
                "rung": set(rungs),
                "hours": {v.strip() for v in args.hours.split(",") if v.strip()},
                "design": {v.strip() for v in args.design.split(",") if v.strip()},
            },
        )
        default_out = path / "packed"
    else:
        if len(rungs) != 1:
            raise SystemExit(
                f"{path} is not a sweep, so name the rung it was solved on with --rung"
            )
        entries = loose_entries(path, rungs[0])
        holder = path.parent if (path / "optimization_results.h5").exists() else path
        default_out = holder / "packed"

    if not entries:
        raise SystemExit("Nothing matched")

    out_dir = Path(args.out) if args.out else default_out
    index = write_all(entries, out_dir, args.bundle)
    print(f"\n{len(entries)} runs, {sum(entry['kb'] for entry in index)} kB")
    print(f"written   : {out_dir}")
    print(f"open      : {out_dir / 'viewer.html'}")


if __name__ == "__main__":
    main()
