# -*- coding: utf-8 -*-
"""Freeze the environment the published results were produced in.

    python scripts/26_environment.py

Writes two files, both from the interpreter that is running, so neither can drift from
what the numbers were computed with:

  requirements.lock.txt   every installed distribution at an exact version, with the
                          interpreter and the MODFLOW 6 build recorded in the header.
                          `make setup` installs from this file, so a fresh clone gets the
                          environment the results came from rather than whatever the
                          version floors in `pyproject.toml` resolve to today.
  docs/LICENCES.md        the licence of every one of those distributions, read from its
                          own metadata.

The project itself is left out of the lock: it is installed from the working tree with
`uv pip install -e . --no-deps`.
"""
from __future__ import annotations

import platform
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "mizan"


def installed() -> list[tuple[str, str, object]]:
    seen, out = set(), []
    for d in distributions():
        name = (d.metadata["Name"] or "").strip()
        key = name.lower().replace("_", "-")
        if not name or key in seen or key == SELF:
            continue
        seen.add(key)
        out.append((name, d.version, d))
    return sorted(out, key=lambda r: r[0].lower())


def modflow_version() -> str:
    """The MODFLOW 6 build in ./bin, which is what every forward run used."""
    exe = ROOT / "bin" / ("mf6.exe" if sys.platform == "win32" else "mf6")
    if not exe.exists():
        return "not installed in ./bin"
    try:
        out = subprocess.run([str(exe), "-v"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+.*)", out)
    return m.group(1).strip() if m else out.strip().splitlines()[0]


# Distributions that ship no licence field, no licence expression and no licence
# classifier in their wheel metadata. The licence is read from the project's own
# repository instead, and the table says so rather than leaving the cell empty.
OFF_METADATA = {
    "google-crc32c": "Apache-2.0, from the project repository",
}

# Anything matching STRONG is strong copyleft and would have to be replaced before this
# repository could ship under the MIT licence it carries. WEAK is file-level copyleft,
# which is compatible with an MIT project that uses the dependency unmodified, and is
# named in the audit rather than folded into "permissive".
STRONG = re.compile(r"\b(GPL-[23]|GPLv[23]|AGPL|LGPL|GNU General Public|"
                    r"GNU Affero|GNU Lesser)", re.I)
WEAK = re.compile(r"\b(MPL|Mozilla Public|EPL|Eclipse Public|CDDL)", re.I)


def licence_of(dist) -> str:
    """One licence string per distribution, preferring the machine-readable fields."""
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr.strip()
    tags = [c.split("::")[-1].strip() for c in (md.get_all("Classifier") or [])
            if c.startswith("License ::")]
    tags = [t for t in tags if t.lower() != "osi approved"]
    if tags:
        return "; ".join(sorted(set(tags)))
    lic = (md.get("License") or "").strip()
    if lic and len(lic) < 60 and "\n" not in lic:
        return lic
    key = (md["Name"] or "").strip().lower().replace("_", "-")
    return OFF_METADATA.get(key, "not declared in the package metadata")


def main() -> None:
    rows = installed()
    header = [
        "# The environment the published results were produced in.",
        "#",
        "# Regenerate with `make env`. Install with `make setup`, which reads this file",
        "# rather than resolving the version floors in pyproject.toml.",
        "#",
        f"# python            {platform.python_version()} ({platform.machine()}, "
        f"{sys.platform})",
        f"# MODFLOW 6         {modflow_version()}",
        f"# distributions     {len(rows)}",
        "#",
        "# MODFLOW 6 and the rest of the USGS executables are fetched by `make setup`",
        "# from the MODFLOW-ORG/executables release. Pass --release-id to",
        "# flopy.utils.get_modflow to pin that download to one release.",
        "",
    ]
    lock = "\n".join(header + [f"{n}=={v}" for n, v, _ in rows]) + "\n"
    (ROOT / "requirements.lock.txt").write_text(lock, encoding="utf-8")

    licences = [(n, v, licence_of(d)) for n, v, d in rows]
    strong = sorted(n for n, _, lic in licences if STRONG.search(lic))
    weak = sorted(n for n, _, lic in licences
                  if WEAK.search(lic) and not STRONG.search(lic))

    if strong:
        verdict = ("**Strong copyleft is present: " + ", ".join(strong) + ".** It has to "
                   "be replaced before this repository can ship under the MIT licence it "
                   "carries.")
    elif weak:
        verdict = ("No dependency carries a GPL, LGPL or AGPL licence. All but "
                   + str(len(weak)) + " are permissive or public domain; "
                   + ", ".join(weak) + " " + ("is" if len(weak) == 1 else "are")
                   + " file-level copyleft under the Mozilla Public Licence 2.0, which "
                     "places obligations on that package's own files and none on the "
                     "code here, which uses "
                   + ("it" if len(weak) == 1 else "them")
                   + " unmodified. Nothing in the stack restricts commercial use or "
                     "piloting.")
    else:
        verdict = ("No dependency carries a GPL, LGPL, AGPL or MPL licence. Every one is "
                   "permissive or public domain, so nothing in the stack restricts "
                   "commercial use or piloting.")

    table = ["# Dependency licence audit", "",
             "Regenerated with `make env` from the metadata of every distribution in the",
             "environment `requirements.lock.txt` pins, so this table and that file",
             "cannot disagree. MODFLOW 6 and the USGS tooling are United States",
             "Government public-domain works.", "", verdict, "",
             "| Distribution | Version | Licence |", "|---|---|---|"]
    for name, version, lic in licences:
        table.append(f"| {name} | {version} | {lic} |")
    table += ["", f"{len(rows)} distributions. MODFLOW 6 {modflow_version()}, US "
                  "Geological Survey, public domain.", ""]
    (ROOT / "docs" / "LICENCES.md").write_text("\n".join(table), encoding="utf-8")

    print(f"requirements.lock.txt: {len(rows)} distributions")
    print(f"docs/LICENCES.md: {len(rows)} rows")


if __name__ == "__main__":
    main()
