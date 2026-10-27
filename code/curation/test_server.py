"""Synthetic-only regression tests; run in the approved execution environment."""

import copy
import http.client
import io
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import Workbench, ReviewError, Conflict, make_handler, on_network_drive
from facad_io import parse_fcd, get_cases_for_image, get_markers, get_image_pixel_spacing_mm
from facad_io import load_review_rows


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / 'images'
        self.data = self.root / 'data'
        self.output = self.root / 'output'
        self.images.mkdir()
        self.data.mkdir()
        self.fcd = self.data / 'd9001.fcd'
        # Interleaved cases ensure tracing numbers are LOCAL to each image.
        # Latin-1 and malformed legacy declarations are intentional parser fixtures.
        self.fcd.write_bytes(('<?xml version="1.0" encoding="broken"?>\n'
            '<facad><patient><name>SYNTHETIC FIXTURE ONLY</name>'
            '<imagefile><idnr>first</idnr><scale>0.1</scale></imagefile>'
            '<imagefile><idnr>second</idnr><scale>0.2</scale></imagefile>'
            '<case><xray_ref>second</xray_ref><author>Fixture Æ</author>'
            '<marker><name>Pg</name><coord>20 30</coord></marker>'
            '<marker><name>PG</name><coord>25 35</coord></marker>'
            '<tooth><marker><name>Isa</name><coord>40 50</coord></marker></tooth>'
            '<marker><name>Ø</name><coord>10 15</coord></marker></case>'
            '<case><xray_ref>first</xray_ref><marker><name>S</name><coord>20 20</coord></marker></case>'
            '<case><xray_ref>second</xray_ref><marker><name>Pg</name><coord>21 31</coord></marker>'
            '<marker><name>Isa</name><coord>240 50</coord></marker></case>'
            '</patient></facad>').encode('latin-1'))
        Image.new('L', (100, 80), 64).save(self.images / 'd9001_1.png')
        Image.fromarray(np.arange(8000, dtype=np.uint16).reshape(80, 100)).save(self.images / 'd9001_2.tiff')
        self.app = Workbench(self.images, self.data, self.output)

    def payload(self, image='d9001_2.tiff', status='accepted', **overrides):
        detail = copy.deepcopy(self.app.detail(image))  # tests edit points in place
        review = detail['review']
        body = {'revision': review['revision'] if review else 0,
                'source_signature': detail['source_signature'], 'status': status,
                'reviewer': 'Synthetic reviewer', 'notes': 'Fixture only',
                'tracing_number': 1, 'points': detail['tracings'][0]['points']}
        body.update(overrides)
        return body

    def test_per_image_numbering_nested_markers_exact_case_and_spacing(self):
        detail = self.app.detail('d9001_2.tiff')
        self.assertEqual([row['number'] for row in detail['tracings']], [1, 2])
        self.assertEqual(detail['tracings'][0]['author'], 'Fixture Æ')
        self.assertEqual(detail['tracings'][0]['points']['Isa'], [40.0, 50.0])
        self.assertEqual(detail['tracings'][0]['points']['Pg'], [20.0, 30.0])
        self.assertEqual(detail['tracings'][0]['points']['PG'], [25.0, 35.0])
        self.assertEqual(detail['pixel_spacing_mm'], 0.2)
        self.assertEqual(detail['tracings'][1]['out_of_bounds'], ['Isa'])
        self.assertEqual(len(self.app.detail('d9001_1.png')['tracings']), 1)

    def test_save_persistence_history_and_stale_tab_conflict(self):
        body = self.payload()
        body['points']['Isa'] = [41.25, 50.5]
        first = self.app.save('d9001_2.tiff', body)
        self.assertEqual(first['revision'], 1)
        self.assertEqual(first['original_points']['Isa'], [40.0, 50.0])
        with self.assertRaises(Conflict):
            self.app.save('d9001_2.tiff', body)
        resumed = Workbench(self.images, self.data, self.output)
        self.assertEqual(resumed.detail('d9001_2.tiff')['review']['points']['Isa'], [41.25, 50.5])
        body['revision'] = 1
        body['status'] = 'draft'
        self.app.save('d9001_2.tiff', body)
        self.assertEqual([row['revision'] for row in self.app.store.history('d9001_2.tiff')], [2, 1])
        self.assertEqual(self.app.store.history('d9001_2.tiff')[1]['status'], 'accepted')

    def test_validation_bounds_empty_points_finite_unknown_names_and_revision(self):
        invalid = [
            {'points': {}}, {'points': {'Pg': [-1, 2]}}, {'points': {'Pg': [100, 2]}},
            {'points': {'Pg': [2, 80]}}, {'points': {'Pg': [float('nan'), 2]}},
            {'points': {'Pg': [float('inf'), 2]}}, {'points': {'Pg': [True, 2]}},
            {'points': {'InventedAnatomy': [2, 2]}}, {'tracing_number': 0},
            {'tracing_number': True}, {'reviewer': '  '}, {'revision': True},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ReviewError):
                self.app.save('d9001_2.tiff', self.payload(**overrides))
        # A draft can preserve bad source points while waiting for expert review.
        body = self.payload(status='draft', tracing_number=2, points={'Isa': [240, 50]})
        self.app.save('d9001_2.tiff', body)

    def test_source_change_blocks_save_and_old_export(self):
        payload = self.payload()
        self.app.save('d9001_2.tiff', payload)
        Image.new('I;16', (100, 80), 15).save(self.images / 'd9001_2.tiff')
        self.assertTrue(self.app.detail('d9001_2.tiff')['stale'])
        payload['revision'] = 1
        with self.assertRaises(Conflict):
            self.app.save('d9001_2.tiff', payload)
        with self.assertRaises(Conflict):
            self.app.export()
        current = self.payload()
        self.app.save('d9001_2.tiff', current)
        with self.app.export() as archive:
            self.assertGreater(len(archive.read()), 0)

    def test_facad_or_image_dimension_change_requires_reindex(self):
        with self.fcd.open('ab') as source:
            source.write(b'\n')
        with self.assertRaises(Conflict):
            self.app.detail('d9001_2.tiff')
        resumed = Workbench(self.images, self.data, self.output)
        Image.new('L', (101, 80), 10).save(self.images / 'd9001_2.tiff')
        with self.assertRaises(Conflict):
            resumed.detail('d9001_2.tiff')

    def test_training_export_roundtrip_second_image_and_unicode(self):
        body = self.payload()
        body['points']['Isa'] = [41.1234567, 51.5]
        self.app.save('d9001_2.tiff', body)
        self.app.save('d9001_1.png', self.payload('d9001_1.png', status='excluded'))
        original_image = (self.images / 'd9001_2.tiff').read_bytes()
        with self.app.export() as file, zipfile.ZipFile(file) as archive:
            self.assertIn('images/d9001_2.tiff', archive.namelist())
            self.assertNotIn('images/d9001_1.png', archive.namelist())
            self.assertEqual(archive.read('images/d9001_2.tiff'), original_image)
            target = self.root / 'roundtrip.fcd'
            target.write_bytes(archive.read('data/d9001.fcd'))
            root = parse_fcd(target)
            self.assertEqual(get_cases_for_image(root, 1), [])
            cases = get_cases_for_image(root, 2)
            self.assertEqual(len(cases), 1)
            points = get_markers(cases[0])
            self.assertAlmostEqual(points['Isa'][0], 41.1234567)
            self.assertEqual(points['Pg'], (20.0, 30.0))
            self.assertEqual(points['PG'], (25.0, 35.0))
            self.assertEqual(points['Ø'], (10.0, 15.0))
            self.assertEqual(get_image_pixel_spacing_mm(root, 2), 0.2)
            self.assertIsNone(root.find('patient/name'))
            selection = self.root / 'roundtrip.csv'
            selection.write_bytes(archive.read('review_selection.csv'))
            self.assertEqual(load_review_rows(selection), [
                ('d9001_1_review.png', 'x', None), ('d9001_2_review.png', '1', [1])])
            audit = json.loads(archive.read('audit.json'))
            self.assertEqual(audit['reviews']['d9001_2.tiff']['tracing_number'], 1)
            self.assertEqual(audit['history']['d9001_2.tiff'][0]['revision'], 1)

    def test_draft_and_excluded_images_never_enter_export(self):
        for status in ('draft', 'excluded'):
            self.app.save('d9001_2.tiff', self.payload(status=status))
            with self.assertRaises(ReviewError):
                self.app.export()

    def test_preview_keeps_native_dimensions_and_does_not_modify_source(self):
        before = (self.images / 'd9001_2.tiff').read_bytes()
        preview = self.app.preview('d9001_2.tiff')
        with Image.open(io.BytesIO(preview)) as image:
            self.assertEqual(image.size, (100, 80))
            self.assertEqual(image.mode, 'L')
            self.assertEqual(image.getextrema(), (0, 255))
        self.assertEqual((self.images / 'd9001_2.tiff').read_bytes(), before)

    def test_old_review_csv_is_a_suggestion_not_automatic_approval(self):
        selection = self.root / 'old.csv'
        selection.write_text('d9001_2_review.png;1,2;\nd9001_1_review.png;x;\n')
        app = Workbench(self.images, self.data, self.output, review_csv=selection)
        self.assertEqual(app.detail('d9001_2.tiff')['imported_selection']['numbers'], [1, 2])
        self.assertIsNone(app.detail('d9001_2.tiff')['review'])
        self.assertTrue(all(row['status'] == 'pending' for row in app.catalog()['items']))

    @unittest.skipUnless(sys.platform == 'win32', 'UNC paths and mapped drives are Windows-only')
    def test_network_paths_avoid_sqlite_wal(self):
        self.assertTrue(on_network_drive(r'\\fileserver.invalid\share\reviews.sqlite3'))
        self.assertFalse(on_network_drive(self.output / 'reviews.sqlite3'))
        with self.app.store.connect() as db:
            self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0], 'wal')

    def test_unlinked_image_can_be_excluded(self):
        text = self.fcd.read_bytes().decode('latin-1').replace('<xray_ref>first</xray_ref>', '<xray_ref>absent</xray_ref>')
        self.fcd.write_bytes(text.encode('latin-1'))
        self.app = Workbench(self.images, self.data, self.output)
        body = {'revision': 0, 'source_signature': self.app.detail('d9001_1.png')['source_signature'],
                'status': 'excluded', 'reviewer': 'Synthetic', 'notes': '', 'tracing_number': None, 'points': {}}
        self.assertEqual(self.app.save('d9001_1.png', body)['status'], 'excluded')

    def test_duplicate_image_index_and_bad_filenames_reported(self):
        Image.new('L', (100, 80)).save(self.images / 'd9001_2.png')
        Image.new('L', (100, 80)).save(self.images / 'arbitrary.png')
        app = Workbench(self.images, self.data, self.output)
        self.assertEqual(len(app.items), 2)
        self.assertTrue(any('duplicate' in issue for issue in app.issues))
        self.assertTrue(any('expected' in issue for issue in app.issues))

    def test_configured_missing_landmarks_can_be_added_without_aliasing(self):
        config = self.root / 'landmarks.yaml'
        config.write_text('landmarks:\n  "N":\n    enabled: true\n    description: Synthetic point\n'
                          '  "Disabled":\n    enabled: false\n')
        self.app = Workbench(self.images, self.data, self.output, landmark_config=config)
        self.assertEqual(self.app.landmarks, {'N': 'Synthetic point'})
        body = self.payload()
        body['points']['N'] = [12.5, 14.5]
        self.assertEqual(self.app.save('d9001_2.tiff', body)['points']['N'], [12.5, 14.5])

    def test_http_origin_host_write_token_and_revision_protection(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.app, 'fixture-token', 'pending'))
        port = server.server_address[1]
        host = f'127.0.0.1:{port}'
        server.RequestHandlerClass = make_handler(self.app, 'fixture-token', host)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def stop():
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
        self.addCleanup(stop)

        def request(path, method='GET', body=None, headers=None):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            try:
                connection.request(method, path, json.dumps(body) if body else None, headers or {})
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        self.assertEqual(request('/')[0], 200)
        self.assertEqual(request('/api/catalog', headers={'Host': 'evil.invalid'})[0], 403)
        self.assertEqual(request('/api/catalog', headers={'Origin': 'https://evil.invalid'})[0], 403)
        self.assertEqual(request('/api/catalog', headers={'Sec-Fetch-Site': 'cross-site'})[0], 403)
        body = dict(self.payload(), image_id='d9001_2.tiff')
        self.assertEqual(request('/api/review', 'POST', body)[0], 403)
        self.assertEqual(request('/api/review', 'POST', body, {'X-Review-Token': 'fixture-token'})[0], 200)
        self.assertEqual(request('/api/review', 'POST', body, {'X-Review-Token': 'fixture-token'})[0], 409)
        self.assertEqual(request('/api/detail?id=../../outside')[0], 400)
        self.assertEqual(request('/../server.py')[0], 404)

    def serve_without_dataset(self):
        defaults = {'output': self.output, 'landmarks': None, 'review_csv': None}
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(None, 'fixture-token', 'pending', defaults))
        host = f'127.0.0.1:{server.server_address[1]}'
        server.RequestHandlerClass = make_handler(None, 'fixture-token', host, defaults)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def stop():
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
        self.addCleanup(stop)

        def request(path, method='GET', body=None, token=True):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=10)
            try:
                headers = {'X-Review-Token': 'fixture-token'} if token else {}
                connection.request(method, path, json.dumps(body) if body is not None else None, headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read() or b'null')
            finally:
                connection.close()
        return request

    def test_folders_can_be_chosen_after_startup_without_a_dataset(self):
        request = self.serve_without_dataset()
        status, catalog = request('/api/catalog')
        self.assertEqual((status, catalog['loaded'], catalog['items']), (200, False, []))
        self.assertEqual(request('/api/detail?id=d9001_1.png')[0], 400)
        both = {'images': f'"{self.images}"', 'facad': str(self.data)}  # pasted Windows paths may be quoted
        self.assertEqual(request('/api/open', 'POST', both, token=False)[0], 403)
        status, opened = request('/api/open', 'POST', dict(both, output=str(self.output)))
        self.assertEqual(status, 200)
        self.assertTrue(opened['loaded'])
        self.assertEqual(sorted(item['id'] for item in opened['items']), ['d9001_1.png', 'd9001_2.tiff'])
        self.assertEqual(opened['folders']['images'], str(self.images.resolve()))
        self.assertEqual(request('/api/detail?id=d9001_1.png')[0], 200)
        # Reopening with a bad folder reports the error and keeps the dataset that is already open.
        self.assertEqual(request('/api/open', 'POST', {'images': str(self.root / 'absent'), 'facad': str(self.data)})[0], 400)
        self.assertEqual(request('/api/open', 'POST', {'images': str(self.images)})[0], 400)
        self.assertTrue(request('/api/catalog')[1]['loaded'])

    def test_browse_lists_folders_and_counts_only(self):
        request = self.serve_without_dataset()
        (self.root / '.hidden').mkdir()
        status, listing = request('/api/browse', 'POST', {'path': str(self.root)})
        self.assertEqual(status, 200)
        self.assertEqual(listing['folders'], ['data', 'images', 'output'])  # dot-folders are hidden
        status, listing = request('/api/browse', 'POST', {'path': str(self.images)})
        self.assertEqual((listing['image_files'], listing['facad_files'], listing['folders']), (2, 0, []))
        self.assertEqual(request('/api/browse', 'POST', {'path': str(self.data)})[1]['facad_files'], 1)
        self.assertEqual(listing['parent'], str(self.root.resolve()))
        self.assertEqual(request('/api/browse', 'POST', {'path': str(self.fcd)})[0], 400)
        self.assertEqual(request('/api/browse', 'POST', {'path': str(self.root / 'absent')})[0], 400)
        self.assertEqual(request('/api/browse', 'POST', {'path': str(self.root)}, token=False)[0], 403)


if __name__ == '__main__':
    unittest.main()
