"""Retrieval of the public Kansas records the L2 rung is scored against.

Three sources, none of which needs a credential:

* **WIMAS** (Kansas Department of Agriculture, Division of Water Resources, served by
  the Kansas Geological Survey). Per-water-right annual **metered** pumping. This is the
  withheld truth: the only abstraction ground truth at this density anywhere.
* **WIZARD** (Kansas Geological Survey). Annual winter water levels at observation and
  irrigation wells.
* **SSEBop** (USGS EROS). Annual actual evapotranspiration for the conterminous United
  States at 1 km, operational and free.

Everything downloaded here is a public record. The WIMAS terms of use forbid resale of
the data or of products derived from it; nothing in this project sells either.
"""
from __future__ import annotations

import http.cookiejar
import io
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

WIMAS = "https://geohydro.kgs.ku.edu/geohydro/wimas/"
WIZARD = "https://geohydro.kgs.ku.edu/geohydro/wizard/"
SSEBOP = ("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/"
          "eta/modis_eta/yearly/downloads/")

# Northwest Kansas, Groundwater Management District 4. Six contiguous counties over the
# Ogallala, chosen before any result was seen: they are the block with both dense
# irrigation and a dense annual water-level network.
GMD4 = {"CN": "Cheyenne", "RA": "Rawlins", "DC": "Decatur",
        "SH": "Sherman", "TH": "Thomas", "SD": "Sheridan"}
# The same six counties as WIZARD numbers them.
WIZARD_COUNTY = {"CN": "23", "RA": "153", "DC": "39",
                 "SH": "181", "TH": "193", "SD": "179"}


# --------------------------------------------------------------------------- transport
def _multipart(fields):
    b = uuid.uuid4().hex
    body = "".join(
        '--{}\r\nContent-Disposition: form-data; name="{}"\r\n\r\n{}\r\n'.format(b, k, v)
        for k, v in fields
    ).encode() + "--{}--\r\n".format(b).encode()
    return b, body


class Session:
    """A cookie-carrying session that has accepted the WIMAS disclaimer."""

    def __init__(self, pause: float = 0.15):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = [("User-Agent", UA)]
        self.pause = pause

    def get(self, url: str, timeout: int = 120) -> str:
        time.sleep(self.pause)
        return self.op.open(url, timeout=timeout).read().decode("latin-1")

    def post(self, url: str, fields, timeout: int = 240) -> str:
        time.sleep(self.pause)
        b, body = _multipart(fields)
        req = urllib.request.Request(
            url, data=body,
            headers={"User-Agent": UA,
                     "Content-Type": "multipart/form-data; boundary=" + b})
        return self.op.open(req, timeout=timeout).read().decode("latin-1")

    def accept_wimas(self) -> None:
        self.op.open(WIMAS, timeout=60).read()
        self.op.open(WIMAS + "query_setup.cfm",
                     data=urllib.parse.urlencode({"wimas_accept": "Accept"}).encode(),
                     timeout=120).read()


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _cells(tr: str) -> list[str]:
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", x)).strip()
            for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]


# --------------------------------------------------------------------------- WIMAS
def _select_county(sess: Session, county: str) -> str:
    """Put a county selection into the session, which pd_list.cfm requires."""
    return sess.post(WIMAS + "query_setup.cfm", [
        ("twp", "0"), ("rng", "0"), ("rng_dir", "W"), ("sect", "0"),
        ("county", "'" + county + "'"), ("umwf", "'IRR'"), ("a_v", "on"),
        ("source", "G"), ("user_email", "gpiw-mizan@localhost"),
        ("Select_wells_button", "Select Water Rights")])


def wimas_points(sess: Session, county: str) -> list[dict]:
    """Every irrigation groundwater point of diversion in one county."""
    sess.accept_wimas()
    first = _select_county(sess, county)

    m = re.search(r"There are ([\d,]+) unique water rights and ([\d,]+) unique points",
                  _text(first))
    n_pd = int(m.group(2).replace(",", "")) if m else 0

    order = urllib.parse.quote("right_type, wr_num")
    pages = [first]
    for start in range(51, n_pd + 51, 50):
        pages.append(sess.get(
            WIMAS + "query_results.cfm?active=off&theStartRow={}&theOrderBy={}".format(
                start, order)))

    out = []
    for page in pages:
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S):
            if "pd_list.cfm" not in tr:
                continue
            c = _cells(tr)
            pd_id = re.search(r"pdiv_id=(\d+)", tr).group(1)
            try:
                lon, lat = float(c[-4]), float(c[-3])
            except (ValueError, IndexError):
                continue
            out.append({"pdiv_id": int(pd_id), "wr": c[0].strip(),
                        "lon": lon, "lat": lat, "county": county})
    seen, uniq = set(), []
    for r in out:
        k = (r["pdiv_id"], r["wr"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


_HIST = re.compile(r'<table border="1".*?</table>', re.S)


def _parse_history(html: str) -> tuple[str, dict]:
    """The water right the page is showing, and its reported annual use in acre-feet."""
    m = re.search(r"Reported Water Use History for Water Right\s*"
                  r"<font[^>]*>([^<]+)</font>", html)
    wr = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    series = {}
    seg = html[html.find("Reported Water Use History"):]
    for tb in _HIST.findall(seg):
        for tr in re.findall(r"<tr>(.*?)</tr>", tb, flags=re.S):
            c = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", x)).strip()
                 for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S)]
            if len(c) == 2 and re.fullmatch(r"\d{4}", c[0]):
                try:
                    series[int(c[0])] = float(c[1].replace(",", ""))
                except ValueError:
                    pass
    return wr, series


def wimas_use(sess: Session, pdiv_id: int) -> dict:
    """Reported annual use for every water right attached to one point of diversion.

    Use is filed per water right, not per well, so the series are keyed by water right
    and de-duplicated by the caller across the points that share one.
    """
    page = sess.get(WIMAS + "pd_list.cfm?pdiv_id={}".format(pdiv_id))
    m = re.search(r'<select[^>]*name="lstWRs"[^>]*>(.*?)</select>', page, flags=re.S)
    rights = (re.findall(r'<option[^>]*value=\s*"?(\d+)"?[^>]*>\s*([^<]*)', m.group(1))
              if m else [])
    uses = re.findall(r'<select[^>]*name="lstUse"[^>]*>(.*?)</select>', page, flags=re.S)
    use_codes = re.findall(r'<option[^>]*value=\s*"?([A-Z]+)"?', uses[0]) if uses else ["IRR"]
    # The page posts back to itself, so every list it carries has to come with the
    # request. The year only drives the detail panel; the history table is complete.
    yrs = re.findall(r'<select[^>]*name="lstWuYrs"[^>]*>(.*?)</select>', page, flags=re.S)
    year = (re.findall(r'value=\s*"?(\d{4})"?', yrs[0]) or ["2025"])[0] if yrs else "2025"

    out = {}
    for wr_id, wr_label in rights:
        label = re.sub(r"\s+", " ", wr_label).strip()
        for use in use_codes:
            html = sess.post(WIMAS + "pd_list.cfm", [
                ("pdiv_id", str(pdiv_id)), ("lstWRs", wr_id), ("lstUse", use),
                ("lstWuYrs", year),
                ("btnGraph", "Graph Water Use History")])
            wr, series = _parse_history(html)
            if series:
                out["{}|{}".format(wr or label, use)] = series
    return out


def fetch_county(county: str, out_dir: Path, workers: int = 4, log=print) -> dict:
    """Points of diversion and per-water-right metered annual use for one county."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "wimas_{}.json".format(county)
    if dest.exists():
        log("  {}: cached".format(county))
        return json.loads(dest.read_text())

    sess = Session()
    pts = wimas_points(sess, county)
    by_wr = {}
    for p in pts:
        by_wr.setdefault(p["wr"], p)
    log("  {}: {} point-of-diversion rows, {} water rights".format(
        county, len(pts), len(by_wr)))

    def job(item):
        wr, p = item
        last = ""
        for attempt in range(3):
            s = Session()
            try:
                s.accept_wimas()
                _select_county(s, county)
                return wr, wimas_use(s, p["pdiv_id"])
            except Exception as exc:                  # network, not logic
                last = repr(exc)[:200]
                time.sleep(2.0 * (attempt + 1))
        return wr, {"_error": last}

    use = {}
    items = list(by_wr.items())
    with ThreadPoolExecutor(workers) as ex:
        for i, (wr, series) in enumerate(ex.map(job, items), 1):
            use[wr] = series
            if i % 100 == 0:
                log("  {}: {}/{} water rights".format(county, i, len(items)))

    rec = {"county": county, "points": pts, "use": use}
    dest.write_text(json.dumps(rec))
    log("  {}: wrote {}".format(county, dest.name))
    return rec


# --------------------------------------------------------------------------- WIZARD
def wizard_wells(sess: Session, gmd: str = "0", county: str = "0",
                 start: str = "01/01/1996",
                 end: str = "12/31/2025") -> list[dict]:
    """Every well in one district or county that carries water levels."""
    html = sess.post(WIZARD + "wizardviewer.cfm", [
        ("f_st", "20"), ("f_c", county),
        ("f_gmd", "0" if gmd == "0" else "'" + gmd + "'"),
        ("f_dstart", start), ("f_dend", end), ("f_only", "Y")])
    m = re.search(r"([\d,]+) records currently selected", _text(html))
    n = int(m.group(1).replace(",", "")) if m else 0

    pages = [html]
    for start_row in range(51, n + 51, 50):
        pages.append(sess.get(
            WIZARD + "wizardwelllisting.cfm?theStartRow={}".format(start_row)))

    out, seen = [], set()
    for page in pages:
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S):
            if "wizardwelldetail.cfm" not in tr:
                continue
            c = _cells(tr)
            wid = re.search(r"usgs_id=(\d+)", tr).group(1)
            if wid in seen:
                continue
            lon = lat = None
            for a, b in zip(c, c[1:]):
                try:
                    fa, fb = float(a), float(b)
                except ValueError:
                    continue
                if -103.5 < fa < -94.0 and 36.5 < fb < 40.5:
                    lon, lat = fa, fb
                    break
            if lon is None:
                continue
            seen.add(wid)
            out.append({"usgs_id": wid, "county": c[1] if len(c) > 1 else "",
                        "lon": lon, "lat": lat})
    return out


_MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def wizard_levels(sess: Session, usgs_id: str) -> dict:
    """Land-surface altitude, well depth, and the dated depth-to-water record.

    Depths are reported in feet below land surface, signed negative in the source.
    They are returned as positive depths so that head equals altitude minus depth.
    """
    head = sess.get(WIZARD + "wizardwelldetail_s.cfm?usgs_id={}".format(usgs_id))
    txt = _text(head)
    alt = re.search(r"Surface Elevation \(ft\):\s*([-\d.]+)", txt)
    dep = re.search(r"Depth of Well \(ft\):\s*([-\d.]+)", txt)
    cty = re.search(r"County:\s*([A-Za-z ]+?)\s+PLSS", txt)

    wl = sess.get(WIZARD + "wizardwelldetail_wl.cfm?usgs_id={}".format(usgs_id))
    levels = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", wl, flags=re.S):
        c = _cells(tr)
        if len(c) < 2:
            continue
        m = re.fullmatch(r"([A-Z]{3})-(\d{2})-(\d{4})", c[0])
        if not m:
            continue
        try:
            v = abs(float(c[1]))
        except ValueError:
            continue
        levels["{}-{:02d}-{}".format(m.group(3), _MON[m.group(1)], m.group(2))] = v
    return {"usgs_id": usgs_id,
            "altitude_ft": float(alt.group(1)) if alt else None,
            "depth_ft": float(dep.group(1)) if dep else None,
            "county": cty.group(1).strip() if cty else "",
            "levels": levels}


def fetch_wizard(out_dir: Path, counties=None, workers: int = 4, log=print) -> dict:
    """Every water-level record in the counties of the study region."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "wizard_levels.json"
    if dest.exists():
        log("  wizard: cached")
        return json.loads(dest.read_text())

    counties = counties or list(GMD4)
    wells, seen = [], set()
    for code in counties:
        s = Session()
        got = wizard_wells(s, county=WIZARD_COUNTY[code])
        for w in got:
            if w["usgs_id"] not in seen:
                seen.add(w["usgs_id"])
                w["county_code"] = code
                wells.append(w)
        log("  wizard: {} {} wells".format(GMD4[code], len(got)))
    log("  wizard: {} wells over the six counties".format(len(wells)))

    def job(w):
        s = Session()
        try:
            rec = wizard_levels(s, w["usgs_id"])
        except Exception as exc:
            return {"usgs_id": w["usgs_id"], "_error": repr(exc)[:200]}
        rec.update(lon=w["lon"], lat=w["lat"], county_code=w.get("county_code", ""))
        return rec

    out = []
    with ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(job, wells), 1):
            out.append(rec)
            if i % 100 == 0:
                log("  wizard: {}/{}".format(i, len(wells)))
    rec = {"counties": counties, "wells": out}
    dest.write_text(json.dumps(rec))
    log("  wizard: wrote {}".format(dest.name))
    return rec


MIRAD = ("https://www.sciencebase.gov/catalog/item/"
         "5db08e84e4b0b0c58b56e04f?format=json")


def fetch_mirad(out_dir: Path, log=print) -> None:
    """The 250 m MIrAD-US irrigated-agriculture maps, 2002, 2007, 2012 and 2017.

    Irrigated extent is what makes a 1 km evapotranspiration pixel unmixable. It is
    licence-and-imagery derived, and carries no water-use information.
    """
    dest = out_dir / "mirad"
    dest.mkdir(parents=True, exist_ok=True)
    if list(dest.rglob("mirad250_17v4.tif")):
        log("  mirad: cached")
        return
    meta = json.loads(urllib.request.urlopen(
        urllib.request.Request(MIRAD, headers={"User-Agent": UA}), timeout=120).read())
    for f in meta["files"]:
        name = f["name"]
        if not name.startswith("mirad250m_") or "(All)" in name:
            continue
        raw = urllib.request.urlopen(
            urllib.request.Request(f["url"], headers={"User-Agent": UA}),
            timeout=900).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extractall(dest)
        log("  mirad: {} {:.1f} MB".format(name, len(raw) / 1e6))


HPSAT = "https://water.usgs.gov/GIS/dsdl/hp_satthk09.zip"


def fetch_hpsat(out_dir: Path, log=print) -> None:
    """The USGS 2009 saturated-thickness grid of the High Plains aquifer, 500 m.

    An ESRI grid in feet on EPSG:5070, published with SIR 2012-5177. It is an
    observation of the aquifer's geometry into which no water-use report enters, which
    is why the layer base is taken from it rather than estimated.
    """
    dest = out_dir / "hpsat"
    if (dest / "hp_satthk09" / "w001001.adf").exists():
        log("  hpsat: cached")
        return
    dest.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(
        urllib.request.Request(HPSAT, headers={"User-Agent": UA}), timeout=900).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall(dest)
    log("  hpsat: {:.1f} MB".format(len(raw) / 1e6))


def ssebop_year(year: int, cache: Path) -> Path:
    """Download and unpack one annual actual-evapotranspiration grid."""
    cache.mkdir(parents=True, exist_ok=True)
    tif = cache / "ssebop_{}.tif".format(year)
    if tif.exists():
        return tif
    raw = urllib.request.urlopen(
        urllib.request.Request(SSEBOP + "y{}.zip".format(year),
                               headers={"User-Agent": UA}), timeout=900).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
        tif.write_bytes(z.read(name))
    return tif


# --------------------------------------------------------------------------- nClimDiv
NCEI = ("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/"
        "time-series/KS-{fips}/pcp/12/12/{y0}-{y1}/data.json")

# Kansas county FIPS codes for the six counties of GMD4 modelled here.
COUNTY_FIPS = {"CN": "023", "RA": "153", "DC": "039",
               "SH": "181", "TH": "193", "SD": "179"}


def fetch_precipitation(out_dir: Path, y0: int = 2000, y1: int = 2024, log=print) -> None:
    """Annual county precipitation from NOAA nClimDiv, via Climate at a Glance.

    The forward model needs an observed driver for recharge. Precipitation over these
    six counties runs from 291 to 674 mm across the record, so a recharge held constant
    in time is falsified by the record before any water-use report is opened. nClimDiv
    is the authoritative United States county series, it carries no water-use term, and
    it needs no credential.
    """
    dest = out_dir / "precip_annual.json"
    if dest.exists():
        log("  precipitation: cached")
        return
    out = {}
    for c, fips in COUNTY_FIPS.items():
        url = NCEI.format(fips=fips, y0=y0, y1=y1)
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180).read()
        d = json.loads(raw)["data"]
        out[c] = {str(y): d["{}12".format(y)]["value"] * 25.4 for y in range(y0, y1 + 1)}
        log("  precipitation {}: {} years".format(c, len(out[c])))
        time.sleep(0.2)
    dest.write_text(json.dumps(out, indent=1))
