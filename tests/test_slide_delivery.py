"""Delivery risks: missing figures, inconsistent values and accidental note edits."""

import base64
import copy

import pytest

from chicago_food.slide_validation import (
    DeckParser,
    ORIGINAL_SLIDES,
    SLIDE_IDS,
    bind_html_metrics,
    note_fingerprints,
    presentation_metric,
    sha256_bytes,
    standalone_html,
    validate_deck,
    validate_notes,
    verify_aggregate_sources,
    write_utf8_atomic,
)


@pytest.fixture
def deck(tmp_path):
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jxZkAAAAASUVORK5CYII=")
    metrics = {"display": {"lr.ap": "0.315", "ablation.history.ap": "0.3146"},
               "values": {"lr.ap": 0.315, "ablation.history.ap": 0.3146},
               "source_refs": {"lr.ap": "test source", "ablation.history.ap": "test source"},
               "format_specs": {"lr.ap": presentation_metric(0.314566, ".3f")[2],
                                "ablation.history.ap": presentation_metric(0.314566, ".4f")[2]},
               "figures": []}
    sections, notes = [], []
    for index, (slide_id, original) in enumerate(zip(SLIDE_IDS, ORIGINAL_SLIDES, strict=True), start=1):
        note = f'<aside class="speaker-note">Original note {original} &amp; café.</aside>' if original else ""
        if note:
            notes.append({"slide_id": slide_id, "original_slide": original, "sha256": sha256_bytes(note.encode())})
        image = ""
        if index <= 9:
            relative = f"reports/slide_figures/{index:02d}.png"
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png)
            image = f'<img src="{relative}" alt="Example top 5 features">'
            metrics["figures"].append({"path": relative, "sha256": sha256_bytes(png)})
        key = f"slide.{index}"
        metrics["display"][key] = f"{index:02d} / 13"
        metrics["values"][key] = metrics["display"][key]
        metrics["source_refs"][key] = "template"
        metrics["format_specs"][key] = presentation_metric(metrics["display"][key], "text")[2]
        sections.append(f'<section id="{slide_id}"><span data-metric="lr.ap">0.315</span>'
                        f'{image}<footer><span data-metric="{key}">{index:02d} / 13</span></footer>{note}</section>')
    sections[0] = sections[0].replace('</section>', '<div data-bar-metric="ablation.history.ap" style="width:89.9%;color:blue"></div></section>')
    document = '<!doctype html><html><head><style>.slide{color:blue}</style></head><body>' + "".join(sections) + '<script>const marker="preserve me";</script></body></html>'
    path = tmp_path / "deck.html"
    path.write_text(document)
    manifest = {"source_commit": "a" * 40, "notes": notes}
    return tmp_path, path, document, metrics, manifest


def check(deck, source=None, *, standalone=False):
    root, path, original, metrics, notes = deck
    return validate_deck(source if source is not None else original, path, root, metrics, notes, standalone=standalone)


def test_complete_deck_has_full_numeric_and_note_coverage(deck):
    result = check(deck)
    assert result["slide_count"] == 13
    assert result["image_count"] == 9
    assert result["metric_binding_coverage"] == 1
    assert result["speaker_notes_preserved"] == 12
    assert result["unbound_visible_numbers"] == 0
    with pytest.raises(ValueError, match="exactly s01 through s13"):
        check(deck, deck[2].replace('</body>', '<section id="s14">Extra slide</section></body>'))


def test_missing_relative_figure_is_rejected(deck):
    (deck[0] / deck[3]["figures"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="Missing local"):
        check(deck)


def test_replaced_figure_bytes_are_rejected(deck):
    (deck[0] / deck[3]["figures"][0]["path"]).write_bytes(b"wrong figure")
    with pytest.raises(ValueError, match="Figure hash"):
        check(deck)


def test_metric_display_mismatch_is_rejected(deck):
    with pytest.raises(ValueError, match="Metric display mismatch"):
        check(deck, deck[2].replace('>0.315<', '>0.999<', 1))


def test_missing_manifest_key_is_rejected(deck):
    del deck[3]["source_refs"]["lr.ap"]
    with pytest.raises(ValueError, match="contract key"):
        check(deck)


def test_changed_numeric_value_is_rejected_before_binding_or_validation(deck):
    deck[3]["values"]["lr.ap"] = 0.999
    with pytest.raises(ValueError, match="values/display format mismatch"):
        bind_html_metrics(deck[2], deck[3])
    with pytest.raises(ValueError, match="values/display format mismatch"):
        check(deck)


def test_changed_display_cannot_be_laundered_by_binding(deck):
    deck[3]["display"]["lr.ap"] = "0.999"
    with pytest.raises(ValueError, match="values/display format mismatch"):
        bind_html_metrics(deck[2], deck[3])
    with pytest.raises(ValueError, match="values/display format mismatch"):
        check(deck)


def test_presentation_rounds_full_source_once_and_retains_units():
    # Pre-rounding AP to .3215 and then to three decimals would incorrectly give .322.
    value, display, _ = presentation_metric(0.321491, ".3f")
    assert value == 0.321 and display == "0.321"
    value, display, spec = presentation_metric(0.225517, ".1%")
    assert value == 0.226 and display == "22.6%" and spec["scale"] == 100
    value, display, spec = presentation_metric(0.1255349, ".1f", scale=100, suffix=" pp")
    assert value == 0.126 and display == "12.6 pp" and spec["suffix"] == " pp"
    assert presentation_metric(-0.00062351, "+.4f")[:2] == (-0.0006, "−0.0006")
    assert presentation_metric(0, "+.4f")[:2] == (0, "+0.0000")
    assert presentation_metric("2025-09-25", "text")[:2] == ("2025-09-25", "2025-09-25")
    assert presentation_metric(55, "d")[:2] == (55, "55")
    assert presentation_metric(float("nan"), ".3f")[:2] == (None, "—")


def test_aggregate_source_hash_rejects_stale_metrics(tmp_path):
    path = tmp_path / "reports/metrics/source.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"ap":0.314566}')
    metrics = {"source_hashes": {"reports/metrics/source.json": sha256_bytes(path.read_bytes())}}
    assert verify_aggregate_sources(tmp_path, metrics) == 1
    path.write_text('{"ap":0.999}')
    with pytest.raises(ValueError, match="Stale slide metrics"):
        verify_aggregate_sources(tmp_path, metrics)


@pytest.mark.parametrize("text", ["Source figure", '<div class="chart-placeholder">Chart placeholder</div>'])
def test_dead_source_figure_and_chart_placeholders_are_rejected(deck, text):
    with pytest.raises(ValueError, match="placeholder"):
        check(deck, deck[2].replace('</body>', text + '</body>'))


def test_six_decimal_values_are_rejected_anywhere_in_source(deck):
    with pytest.raises(ValueError, match="six or more"):
        check(deck, deck[2].replace('>0.315<', '>0.123456<', 1))


def test_note_text_change_is_rejected_even_when_metrics_still_match(deck):
    with pytest.raises(ValueError, match="Speaker-note bytes"):
        check(deck, deck[2].replace('Original note 1 &amp;', 'Altered note 1 &amp;', 1))


def test_unbound_visible_number_is_rejected_but_alt_and_controls_are_not(deck):
    with pytest.raises(ValueError, match="Unbound visible number"):
        check(deck, deck[2].replace('<section id="s01">', '<section id="s01"><p>Year 2025</p>', 1))
    source = deck[2].replace('</body>', '<nav>Zoom 100%</nav></body>')
    assert check(deck, source)["valid"]


def test_ap_bar_width_must_match_rounded_metric_scale(deck):
    with pytest.raises(ValueError, match="AP bar width"):
        check(deck, deck[2].replace('width:89.9%', 'width:99.0%'))


def test_binding_updates_only_leaf_values_and_bars_preserving_note_bytes(deck):
    source = deck[2].replace('>0.315<', '>outdated<').replace('width:89.9%', 'width:0.0%')
    bound = bind_html_metrics(source, deck[3])
    assert bound == deck[2]
    assert note_fingerprints(bound) == note_fingerprints(source)
    assert check(deck, bound)["valid"]


def test_binding_rejects_nested_metric_markup_and_unknown_key(deck):
    with pytest.raises(ValueError, match="leaf span"):
        bind_html_metrics(deck[2].replace('>0.315<', '><b>0.315</b><', 1), deck[3])
    with pytest.raises(ValueError, match="Unknown metric"):
        bind_html_metrics(deck[2].replace('data-metric="lr.ap"', 'data-metric="missing"', 1), deck[3])


def test_standalone_embeds_figures_keeps_css_js_notes_and_rewrites_doc_link(deck):
    root, path, source, metrics, notes = deck
    (root / "README.md").write_text("# Example")
    source = source.replace('</body>', '<a href="README.md#results">Report</a></body>')
    standalone = standalone_html(source, path, root, blob_base="https://github.com/example/repo/blob/feature/slides")
    assert standalone.count('src="data:image/png;base64,') == 9
    assert '<style>.slide{color:blue}</style>' in standalone
    assert '<script>const marker="preserve me";</script>' in standalone
    assert 'href="https://github.com/example/repo/blob/feature/slides/README.md#results"' in standalone
    assert note_fingerprints(standalone) == note_fingerprints(source)
    assert validate_deck(standalone, root / "export.html", root, metrics, notes, standalone=True)["valid"]
    assert path.read_text() == deck[2]


def test_standalone_rejects_tampered_embedded_image(deck):
    root, path, source, metrics, notes = deck
    standalone = standalone_html(source, path, root, blob_base="https://github.com/example/repo/blob/main")
    parsed = DeckParser(standalone)
    image = next(node for node in parsed.elements if node.tag == "img")
    bad = standalone.replace(image.attrs["src"], "data:image/png;base64," + base64.b64encode(b"bad bytes").decode(), 1)
    with pytest.raises(ValueError, match="Embedded figure bytes"):
        validate_deck(bad, root / "export.html", root, metrics, notes, standalone=True)


def test_source_rejects_remote_or_escaping_image_paths(deck):
    for path in ("https://example.com/figure.png", "../figure.png"):
        bad = deck[2].replace(deck[3]["figures"][0]["path"], path, 1)
        with pytest.raises(ValueError, match="relative image|escapes"):
            check(deck, bad)


def test_note_manifest_requires_the_exact_original_slide_assignment(deck):
    manifest = copy.deepcopy(deck[4])
    manifest["notes"][7]["original_slide"] = 8
    with pytest.raises(ValueError, match="original slide mapping"):
        validate_notes(deck[2], manifest)


def test_atomic_writer_preserves_utf8_and_installs_no_temporary_file(tmp_path):
    path = tmp_path / "nested/deck.html"
    write_utf8_atomic(path, "<p>café − 计划</p>")
    assert path.read_bytes().decode() == "<p>café − 计划</p>"
    assert list(path.parent.iterdir()) == [path]
