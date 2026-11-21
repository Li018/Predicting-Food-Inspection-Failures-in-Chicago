"""Bind and validate the existing HTML deck without rewriting speaker notes.

Only local figure bytes and the compact slide metric manifest are needed for
delivery checks. The main model and row-level research artifacts are not loaded.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import html
from html.parser import HTMLParser
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import quote, unquote, urlsplit


SOURCE_HTML_NAME = "Predicting-Food-Inspection-Failures-in-Chicago-EN.html"
SLIDE_IDS = tuple(f"s{number:02d}" for number in range(1, 14))
ORIGINAL_SLIDES = (1, 2, 3, 4, 5, 6, 7, 11, 8, 9, None, 10, 12)
AP_BAR_MAX = 0.35
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
NUMBER = re.compile(r"(?<![A-Za-z0-9_])(?:[+\-−]?\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9_])")
LONG_DECIMAL = re.compile(r"\d+\.\d{6,}")
PLACEHOLDER = re.compile(r"source\s+figure|(?:chart|figure|image|plot)[ -]placeholder|data-placeholder", re.I)
WIDTH = re.compile(r"(?:^|;)\s*width\s*:\s*([+-]?\d+(?:\.\d+)?)%\s*(?=;|$)", re.I)
METRIC_FORMAT = re.compile(r"\+?,?(?:\.[0-4]f|d)")


def _numeric_metric_value(rendered: str, spec: dict) -> int | float:
    number = Decimal(rendered.replace(",", "").replace("−", "-")) / Decimal(str(spec["scale"]))
    if number == number.to_integral_value() and not (number.is_zero() and number.is_signed()):
        return int(number)
    return float(number)


def format_metric_value(value, spec: dict) -> str:
    """Format one canonical presentation value using its declared original unit."""
    if not {"format", "scale", "suffix", "text", "null_text"}.issubset(spec):
        raise ValueError("A metric format specification is incomplete")
    if value is None:
        return spec["null_text"]
    if spec["format"] == "text":
        if not isinstance(value, str) or value != spec["text"]:
            raise ValueError("A text metric differs from its declared text")
        return value
    if not METRIC_FORMAT.fullmatch(spec["format"]) or spec["scale"] not in (1, 100) or spec["text"] is not None:
        raise ValueError("Unsupported numeric metric format specification")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Numeric presentation values must be finite numbers")
    scaled = Decimal(str(value)) * Decimal(str(spec["scale"]))
    if spec["format"].endswith("d"):
        if scaled != scaled.to_integral_value():
            raise ValueError("An integer metric has a fractional presentation value")
        scaled = int(scaled)
    return format(scaled, spec["format"]).replace("-", "−") + spec["suffix"]


def presentation_metric(value, fmt: str, *, scale: int = 1, suffix: str = "") -> tuple:
    """Round the full source once at display precision, retaining original units.

    For example, source AP 0.321491 with .3f becomes value 0.321 and display
    '0.321'. Probability 0.1255 with scale=100/.1f/' pp' becomes value 0.126
    and display '12.6 pp'. Exact source metrics remain in the referenced files.
    """
    if fmt.endswith("%"):
        fmt, scale, suffix = fmt[:-1] + "f", scale * 100, "%" + suffix
    missing = value is None or (not isinstance(value, str) and not math.isfinite(float(value)))
    spec = {"format": fmt, "scale": scale, "suffix": suffix,
            "text": str(value) if fmt == "text" and not missing else None, "null_text": "—"}
    if missing:
        canonical = None
    elif fmt == "text":
        canonical = str(value)
    else:
        # First rounding uses the full source, never the old universal four-digit cache.
        scaled = float(value) * scale
        initial = format(int(scaled) if fmt.endswith("d") else scaled, fmt)
        canonical = _numeric_metric_value(initial, spec)
    return canonical, format_metric_value(canonical, spec), spec


def validate_metric_contract(metrics: dict) -> int:
    """Verify every key, including metrics not currently used in visible spans."""
    names = ("display", "values", "source_refs", "format_specs")
    if any(not isinstance(metrics.get(name), dict) for name in names):
        raise ValueError("Metric contract requires display, values, source_refs and format_specs")
    keys = set(metrics["display"])
    if not keys or any(set(metrics[name]) != keys for name in names[1:]):
        raise ValueError("Metric contract key sets differ")
    for key in sorted(keys):
        value, spec = metrics["values"][key], metrics["format_specs"][key]
        try:
            rendered = format_metric_value(value, spec)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Invalid metric format contract: {key}: {error}") from error
        if rendered != metrics["display"][key]:
            raise ValueError(f"Metric values/display format mismatch: {key}")
        if value is not None and spec["format"] != "text":
            numeric_text = rendered.removesuffix(spec["suffix"]) if spec["suffix"] else rendered
            if value != _numeric_metric_value(numeric_text, spec):
                raise ValueError(f"Metric value is not quantized to its display precision: {key}")
    return len(keys)


def verify_aggregate_sources(root: Path, metrics: dict) -> int:
    """Reject stale slide manifests using only the referenced aggregate files."""
    sources = metrics.get("source_hashes", {})
    if not sources:
        raise ValueError("Slide metrics do not identify their aggregate source hashes")
    for name, digest in sources.items():
        path = (root / name).resolve()
        if not path.is_relative_to((root / "reports/metrics").resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid aggregate source: {name}")
        if sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"Stale slide metrics; aggregate source hash changed: {name}")
    return len(sources)


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None]
    start: int
    open_end: int
    slide_id: str | None
    in_note: bool
    hidden: bool
    close_start: int | None = None
    end: int | None = None
    has_children: bool = False
    text: list[str] = field(default_factory=list)


class DeckParser(HTMLParser):
    """Record source offsets, leaving all original HTML bytes untouched."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0]
        self.line_starts.extend(match.end() for match in re.finditer("\n", source))
        self.stack: list[Element] = []
        self.elements: list[Element] = []
        self.slides: list[str | None] = []
        self.unbound_numbers: list[tuple[str, str]] = []
        self.feed(source)
        self.close()

    def position(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def _start(self, tag, attrs, self_closing=False):
        attributes = dict(attrs)
        parent = self.stack[-1] if self.stack else None
        slide = parent.slide_id if parent else None
        classes = (attributes.get("class") or "").split()
        if tag == "section" and (re.fullmatch(r"s\d+", attributes.get("id") or "") or "slide" in classes):
            slide = attributes.get("id")
            self.slides.append(slide)
        in_note = (parent.in_note if parent else False) or (tag == "aside" and "speaker-note" in classes)
        hidden = (parent.hidden if parent else False) or tag in {"script", "style", "template", "svg"} or "hidden" in attributes
        start = self.position()
        element = Element(tag, attributes, start, start + len(self.get_starttag_text()), slide, in_note, hidden)
        for ancestor in self.stack:
            if "data-metric" in ancestor.attrs:
                ancestor.has_children = True
        self.elements.append(element)
        if self_closing or tag in VOID_TAGS:
            element.close_start = element.open_end
            element.end = element.open_end
        else:
            self.stack.append(element)

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, True)

    def handle_endtag(self, tag):
        match = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i].tag == tag), None)
        if match is None:
            return
        element = self.stack[match]
        element.close_start = self.position()
        element.end = self.source.find(">", element.close_start) + 1
        del self.stack[match:]

    def _text(self, data):
        if not self.stack:
            return
        for ancestor in self.stack:
            if "data-metric" in ancestor.attrs:
                ancestor.text.append(data)
        current = self.stack[-1]
        bound = any("data-metric" in ancestor.attrs for ancestor in self.stack)
        if current.slide_id and not current.in_note and not current.hidden and not bound and NUMBER.search(data):
            self.unbound_numbers.append((current.slide_id, data.strip()[:100]))

    def handle_data(self, data):
        self._text(data)

    def handle_entityref(self, name):
        self._text(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        self._text(html.unescape(f"&#{name};"))

    @property
    def notes(self) -> list[Element]:
        return [node for node in self.elements if node.tag == "aside" and "speaker-note" in (node.attrs.get("class") or "").split()]

    def raw(self, element: Element) -> str:
        if element.end is None:
            raise ValueError(f"Unclosed {element.tag} element on {element.slide_id}")
        return self.source[element.start:element.end]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def note_fingerprints(source: str) -> list[tuple[str | None, str]]:
    parsed = DeckParser(source)
    return [(note.slide_id, sha256_bytes(parsed.raw(note).encode("utf-8"))) for note in parsed.notes]


def validate_notes(source: str, manifest: dict) -> None:
    expected = manifest.get("notes", [])
    mapping = [(slide, original) for slide, original in zip(SLIDE_IDS, ORIGINAL_SLIDES, strict=True) if original is not None]
    if [(record["slide_id"], record["original_slide"]) for record in expected] != mapping:
        raise ValueError("Speaker-note manifest does not preserve the original slide mapping")
    if note_fingerprints(source) != [(record["slide_id"], record["sha256"]) for record in expected]:
        raise ValueError("Speaker-note bytes or their slide assignment changed")


def verify_note_source(root: Path, manifest: dict) -> None:
    """Check the manifest against the original Git source, not only current text."""
    commit = manifest["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Speaker-note source_commit must be a full Git commit SHA")
    result = subprocess.run(["git", "show", f"{commit}:{SOURCE_HTML_NAME}"], cwd=root, capture_output=True)
    if result.returncode:
        raise ValueError("Original speaker-note Git source is unavailable")
    original = DeckParser(result.stdout.decode("utf-8"))
    if len(original.notes) != 12:
        raise ValueError("Original speaker-note source does not contain 12 notes")
    hashes = [sha256_bytes(original.raw(note).encode("utf-8")) for note in original.notes]
    for record in manifest["notes"]:
        if hashes[record["original_slide"] - 1] != record["sha256"]:
            raise ValueError("Speaker-note manifest differs from the original Git source")


def local_path(root: Path, source_path: Path, reference: str) -> Path:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
        raise ValueError(f"A local relative image path is required: {reference[:100]}")
    path = (source_path.parent / unquote(parsed.path)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Local asset path escapes the repository")
    if not path.is_file():
        raise ValueError(f"Missing local image or document: {parsed.path}")
    return path


def expected_bar_width(metrics: dict, key: str) -> float:
    value = metrics["values"].get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= AP_BAR_MAX:
        raise ValueError(f"AP bar metric must be numeric in [0, {AP_BAR_MAX}]: {key}")
    return round(value / AP_BAR_MAX * 100, 1)


def validate_deck(source: str, source_path: Path, root: Path, metrics: dict, notes: dict, *, standalone: bool = False) -> dict:
    """Require image, number, note and figure-byte consistency for all 13 slides."""
    contract_count = validate_metric_contract(metrics)
    if LONG_DECIMAL.search(source):
        raise ValueError("HTML retains a decimal with six or more fractional digits")
    if PLACEHOLDER.search(source):
        raise ValueError("HTML retains a chart placeholder or Source figure link")
    parsed = DeckParser(source)
    if tuple(parsed.slides) != SLIDE_IDS:
        raise ValueError("The deck must contain exactly s01 through s13 in order")
    validate_notes(source, notes)
    display, values, source_refs = (metrics.get(name, {}) for name in ("display", "values", "source_refs"))
    bound_keys, binding_count, bar_count = set(), 0, 0
    for node in parsed.elements:
        if node.in_note:
            continue
        key = node.attrs.get("data-metric")
        if key is not None:
            if node.tag != "span" or node.has_children or node.close_start is None:
                raise ValueError("data-metric must appear on a closed leaf span")
            if key not in display or key not in values or key not in source_refs:
                raise ValueError(f"Missing metric contract key: {key}")
            if "".join(node.text) != str(display[key]):
                raise ValueError(f"Metric display mismatch: {key}")
            binding_count += 1
            bound_keys.add(key)
        bar = node.attrs.get("data-bar-metric")
        if bar is not None:
            width = WIDTH.findall(node.attrs.get("style") or "")
            if len(width) != 1 or float(width[0]) != expected_bar_width(metrics, bar):
                raise ValueError(f"AP bar width disagrees with slide_metrics: {bar}")
            bar_count += 1
        if node.tag == "script" and node.attrs.get("src"):
            raise ValueError("A standalone deck requires inline scripts")
        if node.tag == "link" and "stylesheet" in (node.attrs.get("rel") or "").split():
            raise ValueError("A standalone deck requires inline styles")
    if not binding_count:
        raise ValueError("The deck has no metric bindings")
    if parsed.unbound_numbers:
        slide, text = parsed.unbound_numbers[0]
        raise ValueError(f"Unbound visible number on {slide}: {text}")
    figures = metrics.get("figures", [])
    expected_figures = {figure["path"]: figure for figure in figures}
    if len(figures) != 9 or len(expected_figures) != 9:
        raise ValueError("slide_metrics must identify nine distinct figure paths")
    images = [node for node in parsed.elements if node.tag == "img"]
    if len(images) != 9:
        raise ValueError("The source deck must contain exactly nine figure images")
    observed_paths, observed_hashes = [], []
    for image in images:
        src = image.attrs.get("src") or ""
        if image.attrs.get("srcset"):
            raise ValueError("Figure srcset is unsupported; use the verified src asset")
        if standalone:
            match = re.fullmatch(r"data:image/(?:png|jpeg|svg\+xml);base64,([A-Za-z0-9+/=]+)", src)
            if not match:
                raise ValueError("Standalone images must all be embedded data URIs")
            try:
                content = base64.b64decode(match.group(1), validate=True)
            except ValueError as error:
                raise ValueError("Invalid embedded image") from error
            observed_hashes.append(sha256_bytes(content))
        else:
            path = local_path(root, source_path, src)
            relative = path.relative_to(root.resolve()).as_posix()
            if relative not in expected_figures:
                raise ValueError(f"Image is not listed in slide_metrics: {relative}")
            digest = sha256_bytes(path.read_bytes())
            if digest != expected_figures[relative]["sha256"]:
                raise ValueError(f"Figure hash differs from slide_metrics: {relative}")
            observed_paths.append(relative)
    if standalone:
        if Counter(observed_hashes) != Counter(figure["sha256"] for figure in figures):
            raise ValueError("Embedded figure bytes differ from slide_metrics")
    elif Counter(observed_paths) != Counter(expected_figures.keys()):
        raise ValueError("The deck does not use each of the nine figures exactly once")
    return {"valid": True, "slide_count": len(parsed.slides), "image_count": len(images),
            "speaker_notes_preserved": len(parsed.notes), "metric_binding_count": binding_count,
            "unique_metric_keys": len(bound_keys), "metric_contract_keys_verified": contract_count,
            "metric_binding_coverage": binding_count / (binding_count + len(parsed.unbound_numbers)),
            "unbound_visible_numbers": len(parsed.unbound_numbers), "ablation_bar_count": bar_count,
            "placeholders": 0, "long_decimals": 0, "standalone": standalone,
            "html_sha256": sha256_bytes(source.encode("utf-8"))}


def _replace_attribute(tag: str, attribute: str, value: str) -> str:
    pattern = re.compile(rf"(?<![\w:-]){re.escape(attribute)}\s*=\s*(['\"])(.*?)\1", re.I | re.S)
    match = pattern.search(tag)
    if match is None:
        raise ValueError(f"Expected a quoted {attribute} attribute")
    escaped = html.escape(value, quote=True)
    return tag[:match.start(2)] + escaped + tag[match.end(2):]


def _apply_replacements(source: str, replacements: list[tuple[int, int, str]]) -> str:
    ordered = sorted(replacements)
    if any(left[1] > right[0] for left, right in zip(ordered[:-1], ordered[1:])):
        raise ValueError("Overlapping HTML replacements")
    for start, end, value in reversed(ordered):
        source = source[:start] + value + source[end:]
    return source


def bind_html_metrics(source: str, metrics: dict) -> str:
    """Replace leaf metric text and AP width attributes, preserving every note."""
    validate_metric_contract(metrics)
    parsed, replacements = DeckParser(source), []
    for node in parsed.elements:
        if node.in_note:
            continue
        key = node.attrs.get("data-metric")
        if key is not None:
            if node.tag != "span" or node.has_children or node.close_start is None:
                raise ValueError("data-metric must appear on a closed leaf span")
            if key not in metrics["display"]:
                raise ValueError(f"Unknown metric binding: {key}")
            replacements.append((node.open_end, node.close_start, html.escape(str(metrics["display"][key]), quote=False)))
        bar = node.attrs.get("data-bar-metric")
        if bar is not None:
            width = expected_bar_width(metrics, bar)
            style = node.attrs.get("style") or ""
            if len(WIDTH.findall(style)) != 1:
                raise ValueError("AP bars need one existing width percentage")
            style = WIDTH.sub(lambda match: match.group(0).replace(match.group(1) + "%", f"{width:.1f}%"), style)
            tag = _replace_attribute(source[node.start:node.open_end], "style", style)
            replacements.append((node.start, node.open_end, tag))
    result = _apply_replacements(source, replacements)
    if note_fingerprints(result) != note_fingerprints(source):
        raise RuntimeError("Metric binding modified speaker notes")
    return result


def write_utf8_atomic(path: Path, source: str) -> None:
    """Install a fully validated document without leaving a partial HTML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(source.encode("utf-8"))
    try:
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def github_blob_base(root: Path) -> str:
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
    if remote.startswith("git@github.com:"):
        repository = remote.removeprefix("git@github.com:")
    else:
        parsed = urlsplit(remote)
        if parsed.hostname != "github.com":
            raise ValueError("Repository document links require a GitHub origin")
        repository = parsed.path.lstrip("/")
    repository = repository.removesuffix(".git")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    if not branch:
        branch = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return f"https://github.com/{repository}/blob/{quote(branch, safe='/')}"


def standalone_html(source: str, source_path: Path, root: Path, *, blob_base: str) -> str:
    """Embed verified local figures and make relative documentation links usable."""
    parsed, replacements = DeckParser(source), []
    for node in parsed.elements:
        if node.in_note:
            continue
        if node.tag == "img":
            path = local_path(root, source_path, node.attrs.get("src") or "")
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}.get(path.suffix.lower())
            if mime is None:
                raise ValueError("Unsupported standalone image format")
            value = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            replacements.append((node.start, node.open_end, _replace_attribute(source[node.start:node.open_end], "src", value)))
        elif node.tag == "a" and node.attrs.get("href"):
            href = node.attrs["href"]
            reference = urlsplit(href)
            if reference.scheme or reference.netloc or not reference.path:
                continue
            path = local_path(root, source_path, href)
            relative = path.relative_to(root.resolve()).as_posix()
            value = f"{blob_base.rstrip('/')}/{quote(relative, safe='/')}"
            if reference.query:
                value += "?" + reference.query
            if reference.fragment:
                value += "#" + reference.fragment
            replacements.append((node.start, node.open_end, _replace_attribute(source[node.start:node.open_end], "href", value)))
    result = _apply_replacements(source, replacements)
    if note_fingerprints(result) != note_fingerprints(source):
        raise RuntimeError("Standalone export modified speaker notes")
    return result
