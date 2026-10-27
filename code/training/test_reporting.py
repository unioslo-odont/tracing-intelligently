"""Synthetic reporting checks. Run manually on the HPC; no patient data needed."""

import json
import csv
import os
import tempfile
import unittest

import numpy as np

from metrics import summarize_errors
from report_selection import select_report
from compare_backbones import resolve_reports_dir
from paired_fold_comparison import verdict
from summarize_prediction_tables import summarize_tables
from visualize import resolve_overlay_landmarks


class ReportingTests(unittest.TestCase):
    def test_pooled_median_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'points.csv')
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['serial', 'imagefile', 'landmark', 'error_mm'])
                writer.writeheader()
                for image, value in enumerate([1., 3., 20.]):
                    writer.writerow({'serial': 'synthetic', 'imagefile': str(image),
                                     'landmark': 'A', 'error_mm': value})
            report = summarize_tables([path], ['A'])
            self.assertEqual(report['overall']['median_mm'], 3.)
            self.assertEqual(report['overall']['mre_mm'], 8.)
            with self.assertRaises(ValueError):
                summarize_tables([path, path])

    def test_median_and_invalid_points(self):
        result = summarize_errors(np.array([[1., np.nan], [3., np.inf], [8., np.nan]]), ['A', 'B'])
        self.assertEqual(result['overall']['median_mm'], 3.)
        self.assertEqual(result['overall']['n_points'], 3)
        self.assertIsNone(result['per_landmark']['B']['median_mm'])

    def test_subset_matches_raw_points_with_unequal_coverage(self):
        errors = np.array([[1., 10., 100.], [3., np.nan, 200.]])
        report = summarize_errors(errors, ['A', 'B', 'iGo'])
        selected = select_report(report, ['iGo'])
        expected = summarize_errors(errors[:, :2], ['A', 'B'])
        self.assertEqual(selected['overall']['mre_mm'], expected['overall']['mre_mm'])
        self.assertEqual(selected['overall']['sdr'], expected['overall']['sdr'])
        self.assertEqual(selected['overall']['n_points'], 3)
        self.assertIsNone(selected['overall']['median_mm'])
        self.assertIn('iGo', report['per_landmark'])  # source report is unchanged

    def test_invalid_exclusions_fail(self):
        report = summarize_errors(np.array([[1.]]), ['A'])
        with self.assertRaises(ValueError):
            select_report(report, ['misspelling'])
        with self.assertRaises(ValueError):
            select_report(report, ['A'])

    def test_explicit_report_directory_wins(self):
        with tempfile.TemporaryDirectory() as root:
            fold = os.path.join(root, 'fold0')
            os.makedirs(fold)
            with open(os.path.join(fold, 'eval_val.json'), 'w') as f:
                json.dump({}, f)
            self.assertEqual(resolve_reports_dir(root), root)

    def test_inconclusive_interval_is_not_equivalence(self):
        self.assertIn('not equivalence', verdict(0.02, 0.1))

    def test_overlay_landmarks_are_trained_and_explicit(self):
        trained = ['A', 'B', 'iGo']
        self.assertEqual(resolve_overlay_landmarks(trained), trained)
        self.assertEqual(resolve_overlay_landmarks(trained, ['B', 'A']), ['B', 'A'])
        with self.assertRaises(ValueError):
            resolve_overlay_landmarks(trained, ['untrained'])
        with self.assertRaises(ValueError):
            resolve_overlay_landmarks(trained, ['A', 'A'])


if __name__ == '__main__':
    unittest.main()
