"""
================================================================================
LANDMARK NAME INVENTORY & DUPLICATE-NAME DETECTOR
================================================================================
DESCRIPTION:
    Read-only diagnostic: scans every .fcd file in a directory, collects
    every <marker> name found in EVERY <case> (all tracings, not just the
    one tracing select_reviewed_tracings.py would pick per image - this is
    about the raw naming vocabulary, not which tracing was selected), and
    reports:
      - how often each name appears, and across how many distinct patients
      - whether it's part of the "expected" protocol - read from each
        file's own <cephfile><binding_set> (Facad's own declared landmark
        list for this analysis type), which acts as a ground truth for what
        SHOULD be marked
      - likely duplicate/typo groups: a "#2"-style numbered-duplicate
        suffix, a case-insensitive collision (e.g. 'Is' vs 'IS'), or a
        near-miss spelling of another name already in use

    This exists because the training pipeline (facad_io.get_markers)
    deliberately does NOT canonicalize or deduplicate marker names - any
    inconsistency in the raw Facad data becomes its own landmark channel in
    the trained model (see the "OLp" / "OLp#2" example that motivated this
    script). Run this BEFORE trusting a large landmark vocabulary from
    collect_landmark_vocabulary().

USAGE:
    python analyze_landmark_names.py --fcd-dir /path/to/data/tracings
================================================================================
"""

import argparse
import csv
import difflib
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facad_io import parse_fcd, get_markers

SUFFIX_RE = re.compile(r'^(.*?)\s*#\s*\d+$')


def get_binding_set(root):
    """Return the ordered list of landmark names declared in the first
    <cephfile><binding_set> found, or None if not present."""
    binding_set = root.find('.//cephfile/binding_set')
    if binding_set is None:
        return None
    names = []
    for marker in binding_set.findall('marker'):
        name = marker.findtext('name', '').strip()
        if name:
            names.append(name)
    return names


def scan_directory(fcd_dir):
    """Returns (marker_occurrences, binding_sets_seen).
    marker_occurrences: {name: {'count': int, 'files': set(), 'serials': set()}}
    binding_sets_seen: {tuple(names): [filenames declaring this exact protocol]}"""
    marker_occurrences = defaultdict(lambda: {'count': 0, 'files': set(), 'serials': set()})
    binding_sets_seen = defaultdict(list)

    fcd_files = sorted(f for f in os.listdir(fcd_dir) if f.lower().endswith('.fcd'))
    print(f"Scanning {len(fcd_files)} .fcd file(s) in {fcd_dir} ...")

    for fname in fcd_files:
        path = os.path.join(fcd_dir, fname)
        try:
            root = parse_fcd(path)
        except Exception as e:
            print(f"[ERROR] could not parse {fname}: {e}")
            continue

        patient = root.find('patient')
        serial = patient.findtext('id', fname).strip() if patient is not None else fname

        binding_set = get_binding_set(root)
        if binding_set is not None:
            binding_sets_seen[tuple(binding_set)].append(fname)

        cases = patient.findall('case') if patient is not None else []
        for case in cases:
            for name in get_markers(case).keys():
                entry = marker_occurrences[name]
                entry['count'] += 1
                entry['files'].add(fname)
                entry['serials'].add(serial)

    return marker_occurrences, binding_sets_seen


def find_suffix_variants(names):
    """Names like 'OLp#2' whose base ('OLp') is also a name in use."""
    flags = {}
    name_set = set(names)
    for name in names:
        m = SUFFIX_RE.match(name)
        if m:
            base = m.group(1).strip()
            if base in name_set:
                flags[name] = f"looks like a numbered duplicate of '{base}'"
            else:
                flags[name] = f"has a '#N' suffix but base name '{base}' not otherwise seen"
    return flags


def find_case_collisions(names):
    """Names that only differ by case, e.g. 'Is' vs 'IS'."""
    by_lower = defaultdict(list)
    for name in names:
        by_lower[name.lower()].append(name)
    flags = {}
    for variants in by_lower.values():
        if len(variants) > 1:
            for name in variants:
                others = [v for v in variants if v != name]
                flags[name] = f"differs only by case from: {', '.join(others)}"
    return flags


def find_near_misses(names, already_flagged, cutoff=0.85):
    """Catch-all: names similar (but not identical) to another name, via
    difflib's ratio - flags things like a stray typo that isn't a #N suffix
    or a pure case difference. Skips names already flagged by the more
    specific checks above to avoid noisy double-flagging.

    cutoff=0.85 is deliberately conservative and tested against this
    project's real standard landmark set: cephalometric naming conventions
    routinely use short, similar-looking pairs for genuinely distinct
    adjacent points (Is/Isa, Ii/Iia, Pg/Pgn - tip vs. apex, etc.). At a
    looser cutoff (0.75-0.8) every one of those pairs gets (wrongly)
    flagged, which would drown out the real duplicates this function exists
    to catch. The cost is reduced recall - a single-transposition typo like
    'Nasal' vs 'Nasla' scores 0.8 and slips through at this cutoff. The
    suffix-variant and case-collision checks above are unaffected by this
    tradeoff and should be trusted more than this one."""
    flags = {}
    for name in names:
        if name in already_flagged:
            continue
        others = [n for n in names if n != name]
        candidates = difflib.get_close_matches(name, others, n=3, cutoff=cutoff)
        candidates = [c for c in candidates if c.lower() != name.lower()]  # case collisions handled separately
        if candidates:
            flags[name] = f"similar spelling to: {', '.join(candidates)}"
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fcd-dir', required=True)
    parser.add_argument('--output', default='landmark_name_report.csv')
    args = parser.parse_args()

    marker_occurrences, binding_sets_seen = scan_directory(args.fcd_dir)

    expected_names = set()
    if not binding_sets_seen:
        print("\n[WARN] No <binding_set> found in any file - can't cross-check against an expected protocol.")
    elif len(binding_sets_seen) == 1:
        (protocol_names,) = binding_sets_seen.keys()
        expected_names = set(protocol_names)
        print(f"\nSingle consistent <binding_set> protocol found: {len(protocol_names)} landmarks")
    else:
        print(f"\n[WARN] {len(binding_sets_seen)} DIFFERENT <binding_set> protocols found across files "
              "(different Facad analysis templates were used - expected-protocol check below is skipped):")
        for protocol_names, files in binding_sets_seen.items():
            print(f"  {len(files)} file(s) (e.g. {files[0]}): {len(protocol_names)} landmarks")

    all_names = sorted(marker_occurrences.keys())
    n_total_files = len({f for entry in marker_occurrences.values() for f in entry['files']})

    suffix_flags = find_suffix_variants(all_names)
    case_flags = find_case_collisions(all_names)
    already_flagged = set(suffix_flags) | set(case_flags)
    near_miss_flags = find_near_misses(all_names, already_flagged)

    rows = []
    for name in all_names:
        entry = marker_occurrences[name]
        flag = suffix_flags.get(name) or case_flags.get(name) or near_miss_flags.get(name) or ''
        rows.append({
            'marker_name': name,
            'count': entry['count'],
            'n_files': len(entry['files']),
            'n_patients': len(entry['serials']),
            'in_expected_protocol': name in expected_names if expected_names else '',
            'flag': flag,
        })

    rows.sort(key=lambda r: -r['count'])

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['marker_name', 'count', 'n_files', 'n_patients', 'in_expected_protocol', 'flag'])
        writer.writeheader()
        writer.writerows(rows)

    n_flagged = sum(1 for r in rows if r['flag'])
    n_nonstandard = sum(1 for r in rows if expected_names and not r['in_expected_protocol'])

    print(f"\nFound {len(all_names)} distinct marker names across {n_total_files} file(s).")
    if expected_names:
        print(f"{n_nonstandard} name(s) not in the expected protocol.")
    print(f"{n_flagged} name(s) flagged as likely duplicates/typos.\n")

    print(f"{'Name':<20} {'Count':<7} {'Files':<7} {'Patients':<9} {'Expected':<9} Flag")
    for r in rows:
        is_interesting = r['flag'] or (expected_names and not r['in_expected_protocol'])
        if is_interesting:
            expected_str = 'yes' if r['in_expected_protocol'] else ('NO' if expected_names else '?')
            print(f"{r['marker_name']:<20} {r['count']:<7} {r['n_files']:<7} {r['n_patients']:<9} "
                  f"{expected_str:<9} {r['flag']}")

    print(f"\nFull report (all {len(all_names)} names): {args.output}")


if __name__ == "__main__":
    main()
