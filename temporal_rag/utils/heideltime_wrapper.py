from __future__ import annotations

import calendar
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from ..paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

_PY_HEIDELTIME_SEQUENTIAL_PATCHED = False


def _default_heideltime_treetagger_home() -> str | None:
    """Use repo-local TreeTagger install when present (no env var required)."""
    cand = PROJECT_ROOT / "vendor" / "treetagger-install"
    if (cand / "bin" / "tree-tagger").is_file():
        return str(cand.resolve())
    return None


def _patch_py_heideltime_treetagger_home() -> None:
    """
    If HEIDELTIME_TREETAGGER_HOME is set (or ``vendor/treetagger-install`` exists), rewrite
    py_heideltime's config.props writer so Java uses that TreeTagger installation instead of the
    bundled Linux path (wrong binaries on macOS).
    """
    home = (os.environ.get("HEIDELTIME_TREETAGGER_HOME") or "").strip()
    if not home:
        home = (_default_heideltime_treetagger_home() or "").strip()
        if home:
            os.environ.setdefault("HEIDELTIME_TREETAGGER_HOME", home)
    if not home:
        return
    try:
        import py_heideltime.config as phtc  # type: ignore[import]
    except ImportError:
        return

    tagger_path = str(Path(home).expanduser().resolve())
    lib = Path(phtc.__file__).resolve().parent
    template_path = lib / "resources" / "config_props_template"

    def _write_config_props_override() -> None:
        conf_template = template_path.read_text(encoding="utf-8")
        Path("config.props").write_text(
            conf_template.replace("{path}", tagger_path),
            encoding="utf-8",
        )

    phtc._write_config_props = _write_config_props_override  # type: ignore[assignment]
    logger.info("HeidelTime TreeTagger path overridden: %s", tagger_path)


def _patch_py_heideltime_sequential() -> None:
    """
    Replace ``py_heideltime.heideltime`` (multiprocessing Pool) with sequential JVM calls.

    macOS uses the ``spawn`` multiprocessing start method; Pool workers typically lack the parent
    cwd and ``config.props``, so HeidelTime cannot find TreeTagger (Stream closed / no sentence tokens).
    """
    global _PY_HEIDELTIME_SEQUENTIAL_PATCHED
    if _PY_HEIDELTIME_SEQUENTIAL_PATCHED:
        return
    try:
        import py_heideltime as ph_root  # type: ignore[import]
        import py_heideltime.py_heideltime as pht  # type: ignore[import]
    except ImportError:
        return

    def heideltime_sequential(
        text: str,
        language: str = "english",
        document_type: str = "news",
        dct: str | None = None,
    ) -> list:
        from py_heideltime.config import _write_config_props  # type: ignore[import]
        from py_heideltime.utils import process_text  # type: ignore[import]

        pht._validate_inputs(language, document_type)
        _write_config_props()
        try:
            with tempfile.TemporaryDirectory(dir=pht.LIBRARY_PATH) as tempdir:
                processed_text = process_text(text)
                filepaths = pht._create_text_files(processed_text, tempdir)
                tml_docs = [
                    pht._exec_java_heideltime(str(fp), language, document_type, dct)
                    for fp in filepaths
                ]
                tml_content = "".join(
                    re.findall(r"<TimeML>(.*)</TimeML>", doc, re.DOTALL)[0].strip("\n")
                    for doc in tml_docs
                )
                return pht._get_timexs(tml_content)
        finally:
            try:
                Path("config.props").unlink()
            except FileNotFoundError:
                pass

    pht.heideltime = heideltime_sequential  # type: ignore[assignment]
    ph_root.heideltime = heideltime_sequential  # type: ignore[assignment]
    _PY_HEIDELTIME_SEQUENTIAL_PATCHED = True


# ──────────────────────────────────────────────────────────────────────────────
# ISO-8601 helpers
# ──────────────────────────────────────────────────────────────────────────────


def _complete_iso_date(value: str, as_end: bool = False) -> str | None:
    """
    Convert partial ISO dates to full YYYY-MM-DD strings.

    HeidelTime sometimes returns "2016", "2016-10", "2016-W42", etc.
    We expand these to a concrete calendar date so epoch conversion works.

    Parameters
    ----------
    value   : ISO-8601 string from HeidelTime (may be partial).
    as_end  : If True, return the *last* day of the interval (e.g. month-end).
              If False, return the *first* day.
    """
    if not value:
        return None
    v = value.strip()
    # Reject any HeidelTime meta-values that aren't real dates
    if v.upper() in {"UNDEF", "PRESENT_REF", "PRESEN", "PAST_REF", "PAS",
                     "FUTURE_REF", "FU"} or v.startswith("UNDEF") or "X" in v:
        return None

    # Already a full date
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return v

    # Year-only  "2016"
    if len(v) == 4 and v.isdigit():
        return f"{v}-12-31" if as_end else f"{v}-01-01"

    # Year-Month  "2016-10"  (skip season codes like "2016-SU", "2016-FA", etc.)
    if len(v) == 7 and v[4] == "-":
        suffix = v[5:7]
        if not suffix.isdigit():
            # Season codes: SU=summer, FA=fall, WI=winter, SP=spring
            season_month = {"SP": ("03", "05"), "SU": ("06", "08"),
                            "FA": ("09", "11"), "WI": ("12", "02")}
            if suffix in season_month:
                m_start, m_end = season_month[suffix]
                return f"{v[:4]}-{m_end}-30" if as_end else f"{v[:4]}-{m_start}-01"
            return None
        year, month = int(v[:4]), int(suffix)
        if as_end:
            last_day = calendar.monthrange(year, month)[1]
            return f"{v}-{last_day:02d}"
        return f"{v}-01"

    # ISO week  "2016-W42"
    if len(v) == 8 and v[5] == "W":
        try:
            year, week = int(v[:4]), int(v[6:8])
            # ISO week Monday = day 1; use Friday (5) for end
            day = 1 if not as_end else 7
            dt = datetime.fromisocalendar(year, week, day)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            pass

    # Quarter  "2016-Q3"
    if len(v) == 7 and v[5] == "Q":
        quarter = int(v[6])
        month_start = (quarter - 1) * 3 + 1
        if as_end:
            month_end = quarter * 3
            last_day = calendar.monthrange(int(v[:4]), month_end)[1]
            return f"{v[:4]}-{month_end:02d}-{last_day:02d}"
        return f"{v[:4]}-{month_start:02d}-01"

    # Datetime  "2016-10-18T12:00"  — truncate to date
    if "T" in v:
        return v.split("T")[0]

    return None


def iso_to_epoch(date_str: str) -> int | None:
    """Convert a YYYY-MM-DD string to a UTC Unix epoch (seconds). Returns None on failure."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────────


class HeidelTimeWrapper:
    """Wrapper for the HeidelTime temporal text processing engine."""

    def __init__(self, language: str = "English", document_type: str = "news"):
        """
        Parameters
        ----------
        language      : Text language passed to HeidelTime.
        document_type : HeidelTime document type: "news", "narrative",
                        "colloquial", "scientific".
        """
        self.language = language
        self.document_type = document_type

    # ── Raw extraction ────────────────────────────────────────────────────────

    def extract_resolved_dates(
        self,
        text: str,
        document_creation_time: str,
        doc_type: str | None = None,
    ) -> list[str]:
        """
        Run HeidelTime on *text* and return sorted resolved ``YYYY-MM-DD`` strings.

        Year / month / day granularity comes from HeidelTime TIMEX ``value`` fields,
        then ``_complete_iso_date`` expands partial ISO to concrete endpoints.

        Parameters
        ----------
        text                    : Text to process.
        document_creation_time  : DCT anchor for relative expressions (YYYY-MM-DD).
        doc_type                : Override instance document_type for this call.

        Returns
        -------
        list[str]  Sorted list of resolved "YYYY-MM-DD" dates (may be empty).
        """
        _patch_py_heideltime_treetagger_home()
        _patch_py_heideltime_sequential()

        try:
            import py_heideltime.py_heideltime as pht  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("py_heideltime not installed: pip install py_heideltime") from exc

        if not text or not text.strip():
            return []

        try:
            results = pht.heideltime(
                text,
                language=self.language,
                document_type=doc_type or self.document_type,
                dct=document_creation_time,
            )
        except Exception as exc:
            logger.warning("HeidelTime failed on input %r: %s", text[:60], exc)
            return []

        resolved: list[str] = []
        for item in results:
            if item.get("type") != "DATE":
                continue
            val = item.get("value", "")
            if not val or val.startswith("UNDEF"):
                continue
            start_d = _complete_iso_date(val, as_end=False)
            end_d = _complete_iso_date(val, as_end=True)
            if start_d:
                resolved.append(start_d)
            if end_d and end_d != start_d:
                resolved.append(end_d)

        return sorted(set(resolved))

    def extract_temporal_bounds(
        self, text: str, document_creation_time: str
    ) -> Tuple[str, str]:
        """
        Extract T_start / T_end from a text chunk.

        Returns
        -------
        (T_start, T_end) as ISO-8601 strings. Falls back to
        document_creation_time if no dates are found.
        """
        dates = self.extract_resolved_dates(text, document_creation_time)
        if not dates:
            return document_creation_time, document_creation_time
        return dates[0], dates[-1]

    def parse_query_range(
        self, query: str, reference_date: str
    ) -> Tuple[str, str]:
        """
        Parse a natural-language query and extract temporal search boundaries.

        Returns
        -------
        (q_start, q_end) as ISO-8601. Falls back to reference_date.
        """
        dates = self.extract_resolved_dates(
            query, reference_date, doc_type="colloquial"
        )
        if not dates:
            return reference_date, reference_date
        return dates[0], dates[-1]
