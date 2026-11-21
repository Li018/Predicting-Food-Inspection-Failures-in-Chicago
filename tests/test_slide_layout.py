"""Protect the 13-slide structure, usable figure paths and original authored notes."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
HTML = "Predicting-Food-Inspection-Failures-in-Chicago-EN.html"
SOURCE_SHA256 = "3ce0061daefa485f6e887d014449b4ea8671d96e1b60045218785d038766ed26"
ORIGINAL_NOTE_HASHES = {
    1: "393aab963ebcc4d732d3e6c8d14f27cf15c0f20a34d5f546b81e8972dca4f0bd",
    2: "6bcd828756de47b7b96009c7b74989b2b54b1d437b4b1c8d46bb4953300d4d6c",
    3: "f4588703340b5c5e7b1831d8787a06a0caf971cc2235072f9b92c9349c959396",
    4: "49e5cc4bf665e5a054c3f891412cda8dcb9672eaf689f52d7490200dd6400f33",
    5: "ac98925723e4b90277dc764d22c043b792f86761c71449c86a91c7d992ce728e",
    6: "53bb7f18708348b8f73a2c8607728d791630401c47b977004ead76348031dd08",
    7: "cf0326069c0cdd391c7d79483a4c6c2edab6f0cafbeda2300861cca20d868374",
    8: "083032ff4304c53837d433e275a6aef689afda0e506a80791c7cd017667c57ef",
    9: "376f82c55475c42bf74bc994181ebce3d604a07258f0567b6c259def4e8e95eb",
    10: "81ee995af2a4ba58eb9bebf82f8e116f3f810ce361c0a5aaec40905453e24588",
    11: "d89291eec04f0d2444ec4e7a7e488ca4462c7273800d21f186f8b45b3b3d0bb0",
    12: "ecaa54288fa259b2e8eb2590534050c380a2369cadcf228690dcc41e613571cb",
}
NOTE_SOURCE_ORDER = [1, 2, 3, 4, 5, 6, 7, 11, 8, 9, None, 10, 12]
EXPECTED_FIGURES = {
    3: ["01_seasonality_drift.png"],
    6: ["02_precision_recall.png"],
    7: ["03_calibration.png"],
    8: ["04_daily_capacity.png"],
    9: ["05_grouped_shap.png", "06_shap_dependence.png"],
    10: ["07_error_fp.png", "08_error_fn.png"],
    11: ["09_subgroup_audit.png"],
}


class Tags(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.items = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


def elements(source, tag=None, class_name=None):
    return [attrs for name, attrs in Tags(source).items
            if (tag is None or name == tag)
            and (class_name is None or class_name in attrs.get("class", "").split())]


def sections():
    text = (ROOT / HTML).read_text()
    content = text.split("<!-- DECK CONTENT START -->", 1)[1].split("<!-- DECK CONTENT END -->", 1)[0]
    return text, re.findall(r'<section\b[^>]*class="[^\"]*\bslide\b[^\"]*"[^>]*>.*?</section>', content, re.S)


def test_thirteen_unique_slides_have_the_requested_business_audit_pilot_order():
    _, slides = sections()
    assert len(slides) == 13
    descriptors = [elements(slide, "section")[0] for slide in slides]
    assert [item["id"] for item in descriptors] == [f"s{i:02d}" for i in range(1, 14)]
    assert all(item.get("data-title", "").strip() for item in descriptors)
    assert sum("active" in item["class"].split() for item in descriptors) == 1
    for number, layout in {8: "capacity-layout", 9: "explanation-layout", 10: "case-grid",
                           11: "fairness-layout", 12: "limits-layout", 13: "pilot-steps"}.items():
        assert len(elements(slides[number - 1], class_name=layout)) == 1


def test_nine_figure_assets_are_real_relative_images_on_the_correct_slides():
    _, slides = sections()
    total = 0
    for number, slide in enumerate(slides, 1):
        images = elements(slide, "img")
        expected = EXPECTED_FIGURES.get(number, [])
        assert [image["src"] for image in images] == [f"reports/slide_figures/{name}" for name in expected]
        for image in images:
            total += 1
            assert len(image.get("alt", "").strip()) >= 10
            path = ROOT / image["src"]
            assert path.is_file()
            assert path.resolve().is_relative_to((ROOT / "reports/slide_figures").resolve())
            header = path.read_bytes()[:24]
            assert header[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = struct.unpack(">II", header[16:24])
            assert width >= 1000 and height >= 500
    assert total == 9


def test_five_act_links_resolve_through_the_preserved_presentation_hash_router():
    source, slides = sections()
    # The supplied presentation uses #slide-N routes, while section IDs are sNN.
    # Do not incorrectly require every navigation link to match a DOM ID literally.
    assert re.search(r"addEventListener\(['\"]hashchange['\"],\s*fromHash\)", source)
    assert re.search(r"function fromHash\(\).*?location\.hash\.match", source, re.S)
    expected_targets = [1, 3, 6, 9, 12]
    for number, slide in enumerate(slides[1:], 2):
        navs = re.findall(r'<nav\b[^>]*class="act-nav"[^>]*>.*?</nav>', slide, re.S)
        assert len(navs) == 1
        anchors = elements(navs[0], "a")
        assert len(anchors) == 5
        destinations = [int(re.fullmatch(r"#slide-(\d+)", item["href"]).group(1)) for item in anchors]
        assert destinations == expected_targets
        assert all(1 <= target <= len(slides) for target in destinations)
        current = [i for i, item in enumerate(anchors) if item.get("aria-current") == "step"]
        assert current == [max(i for i, target in enumerate(expected_targets) if target <= number)]


def test_original_twelve_complete_speaker_note_blocks_keep_their_exact_hashes():
    _, slides = sections()
    manifest = json.loads((ROOT / "reports/metrics/slide_notes_manifest.json").read_text())
    assert manifest["source_commit"] == "176c1ebdc2f1856cd209ca31ff029806106a8e74"
    assert manifest["source_html_sha256"] == SOURCE_SHA256
    assert len(manifest["notes"]) == 12
    recorded = {item["slide_id"]: item for item in manifest["notes"]}
    assert len(recorded) == 12
    assert {item["original_slide"] for item in recorded.values()} == set(ORIGINAL_NOTE_HASHES)
    for number, (slide, original_number) in enumerate(zip(slides, NOTE_SOURCE_ORDER), 1):
        blocks = re.findall(r'<aside class="speaker-note">.*?</aside>', slide, re.S)
        slide_id = f"s{number:02d}"
        if original_number is None:
            assert blocks == [] and slide_id not in recorded
        else:
            assert len(blocks) == 1
            digest = hashlib.sha256(blocks[0].encode()).hexdigest()
            assert digest == ORIGINAL_NOTE_HASHES[original_number]
            assert recorded[slide_id]["original_slide"] == original_number
            assert recorded[slide_id]["sha256"] == digest


def test_chart_placeholders_and_source_figure_dead_links_are_removed():
    source, slides = sections()
    assert not re.search(r"(?:chart|figure|image|logo)[-_ ]placeholder", source, re.I)
    assert not re.search(r"Source\s+figure", source, re.I)
    for slide in slides:
        for anchor in elements(slide, "a"):
            destination = anchor.get("href", "")
            assert destination and destination != "#"
            if "://" not in destination and not destination.startswith("#"):
                assert (ROOT / destination.split("#", 1)[0]).exists()


def test_cover_checks_ablation_and_roadmap_use_accessible_native_components():
    _, slides = sections()
    cover_tags = re.search(r'<div class="cover-tags">.*?</div>', slides[0], re.S).group()
    icons = elements(cover_tags, "svg")
    assert len(icons) == 3
    assert all(icon.get("aria-hidden") == "true" and icon.get("viewbox") == "0 0 24 24" for icon in icons)
    checks = re.search(r'<div class="check-grid">.*?</div>\s*</div>', slides[3], re.S).group()
    assert len(elements(checks, "article")) == 4
    assert len(elements(checks, "svg")) == 4
    assert len(elements(slides[4], "article", "feature-card")) == 3
    assert len(elements(slides[4], class_name="ablation-row")) == 3
    assert len(elements(slides[5], "tr", "selected-model")) == 1
    assert len(elements(slides[12], "article", "pilot-step")) == 3
    assert len(elements(slides[12], class_name="step-arrow")) == 2
