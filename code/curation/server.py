"""Local Facad review workbench. No torch, hosted services, or patient uploads."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import sqlite3
import sys
import tempfile
import threading
import webbrowser
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data_preparation_TSD'))
from facad_io import parse_fcd, get_cases_for_image, get_markers, get_image_pixel_spacing_mm
from facad_io import load_review_rows, resolve_serial_and_image_index

STATIC = Path(__file__).with_name('static')
IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png'}
IMAGE_FILENAME = re.compile(r'^[A-Za-z]\d+_[1-9]\d*\.(?:tif|tiff|png)$', re.IGNORECASE)


class ReviewError(ValueError):
    pass


class Conflict(ReviewError):
    pass


def digest(path):
    # Chunked rather than hashlib.file_digest, which needs Python 3.11.
    sha = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def on_network_drive(path):
    """True for Windows UNC paths and mapped network drives. Never touches the filesystem."""
    if sys.platform != 'win32':
        return False
    text = str(Path(path).absolute())
    if text.startswith('\\\\'):
        return True
    import ctypes
    drive = Path(text).drive
    return bool(drive) and ctypes.windll.kernel32.GetDriveTypeW(drive + '\\') == 4  # DRIVE_REMOTE


class ReviewStore:
    """Append-only revision history plus an atomically updated current review."""

    def __init__(self, path, dataset_id):
        self.path, self.dataset_id = Path(path), dataset_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            # WAL relies on shared memory that network filesystems do not provide reliably.
            db.execute(f"PRAGMA journal_mode={'DELETE' if on_network_drive(self.path) else 'WAL'}")
            db.execute('CREATE TABLE IF NOT EXISTS reviews ('
                       'dataset TEXT, image TEXT, revision INTEGER, body TEXT, '
                       'PRIMARY KEY(dataset, image))')
            db.execute('CREATE TABLE IF NOT EXISTS history ('
                       'dataset TEXT, image TEXT, revision INTEGER, body TEXT, '
                       'PRIMARY KEY(dataset, image, revision))')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def all(self):
        with self.connect() as db:
            return {row[0]: json.loads(row[1]) for row in db.execute(
                'SELECT image, body FROM reviews WHERE dataset=?', (self.dataset_id,))}

    def get(self, image):
        with self.connect() as db:
            row = db.execute('SELECT body FROM reviews WHERE dataset=? AND image=?',
                             (self.dataset_id, image)).fetchone()
            return json.loads(row[0]) if row else None

    def history(self, image):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT body FROM history WHERE dataset=? AND image=? ORDER BY revision DESC',
                (self.dataset_id, image))]

    def save(self, image, expected_revision, review):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM reviews WHERE dataset=? AND image=?',
                             (self.dataset_id, image)).fetchone()
            current = row[0] if row else 0
            if current != expected_revision:
                raise Conflict('This image was saved in another tab. Reload it before saving again.')
            review = dict(review, revision=current + 1,
                          saved_at=datetime.now(timezone.utc).isoformat())
            body = encode_json(review)
            db.execute('INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?)',
                       (self.dataset_id, image, current + 1, body))
            db.execute('INSERT INTO history VALUES (?, ?, ?, ?)',
                       (self.dataset_id, image, current + 1, body))
        return review


class Workbench:
    def __init__(self, images_dir, fcd_dir, output_dir, landmark_config=None,
                 review_csv=None, demo=False):
        self.images_dir = Path(images_dir).resolve()
        self.fcd_dir = Path(fcd_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if not self.images_dir.is_dir() or not self.fcd_dir.is_dir():
            raise ReviewError('The images and Facad directories must both exist.')
        self.demo = demo
        self.landmarks = {}
        if landmark_config:
            import yaml
            with Path(landmark_config).open(encoding='utf-8') as source:
                vocabulary = yaml.safe_load(source).get('landmarks', {})
            self.landmarks = {str(name): metadata.get('description', '')
                              for name, metadata in vocabulary.items() if metadata.get('enabled')}
        self.imported = {}
        if review_csv:
            for filename, raw, numbers in load_review_rows(review_csv):
                serial, index = resolve_serial_and_image_index(filename)
                self.imported[(serial, index)] = {'raw': raw, 'numbers': numbers}
        identity = f'{self.images_dir}\n{self.fcd_dir}'
        self.dataset_id = hashlib.sha256(identity.encode()).hexdigest()
        self.store = ReviewStore(self.output_dir / 'reviews.sqlite3', self.dataset_id)
        self.items, self.issues = {}, []
        self._index()

    def _index(self):
        roots = {}
        seen = set()
        fcd_files = {p.stem: p for p in self.fcd_dir.iterdir() if p.is_file() and p.suffix.lower() == '.fcd'}
        for path in sorted(self.images_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            serial, index = resolve_serial_and_image_index(path.name)
            if not IMAGE_FILENAME.fullmatch(path.name):
                self.issues.append(f'{path.name}: expected <letter><digits>_<image_index>.png/.tif/.tiff '
                                   '(the training loader naming convention).')
                continue
            if (serial, index) in seen:
                self.issues.append(f'{path.name}: duplicate image index; only the first file is indexed.')
                continue
            fcd_path = fcd_files.get(serial)
            if fcd_path is None:
                self.issues.append(f'{path.name}: no matching Facad file.')
                continue
            try:
                if serial not in roots:
                    roots[serial] = (parse_fcd(fcd_path), digest(fcd_path))
                root, fcd_hash = roots[serial]
                with Image.open(path) as image:
                    width, height = image.size
                patient = root.find('patient')
                image_nodes = patient.findall('imagefile') if patient is not None else []
                if index > len(image_nodes):
                    raise ReviewError('Image index is absent from the Facad file.')
                tracings = []
                for number, case in enumerate(get_cases_for_image(root, index) or [], 1):
                    raw = get_markers(case)
                    points = {name: list(xy) for name, xy in raw.items()
                              if all(math.isfinite(v) for v in xy)}
                    warnings = []
                    if len(points) != len(raw):
                        warnings.append('Non-finite coordinates omitted.')
                    for marker in case.findall('.//marker'):
                        name = marker.findtext('name', '').strip()
                        if name and name not in points:
                            warnings.append(f'{name}: unreadable coordinates.')
                    names = [m.findtext('name', '').strip() for m in case.findall('.//marker')]
                    duplicates = sorted({name for name in names if name and names.count(name) > 1})
                    if duplicates:
                        warnings.append('Repeated marker names (last coordinate used): ' + ', '.join(duplicates))
                    author = next((case.findtext(tag, '').strip() for tag in
                                   ('author', 'operator', 'createdby', 'user')
                                   if case.findtext(tag, '').strip()), '')
                    tracings.append({'number': number, 'author': author,
                                     'name': case.findtext('name', '').strip(), 'points': points,
                                     'marker_names': sorted(set(name for name in names if name)),
                                     'warnings': warnings, 'out_of_bounds': self.outside(points, width, height)})
                spacing = get_image_pixel_spacing_mm(root, index)
                if spacing is not None and (not math.isfinite(spacing) or spacing <= 0):
                    spacing = None
                self.items[path.name] = {
                    'id': path.name, 'serial': serial, 'image_index': index,
                    'width': width, 'height': height, 'pixel_spacing_mm': spacing,
                    'tracings': tracings, 'image_path': path, 'fcd_path': fcd_path,
                    'fcd_hash': fcd_hash, 'image_count': len(image_nodes),
                    'scales': [node.findtext('scale', '').strip() for node in image_nodes],
                }
                seen.add((serial, index))
            except Exception as error:
                self.issues.append(f'{path.name}: {error}')

    @staticmethod
    def outside(points, width, height):
        return [name for name, (x, y) in points.items() if not (0 <= x < width and 0 <= y < height)]

    def item(self, image_id):
        if image_id not in self.items:
            raise ReviewError('Image is not in this dataset.')
        return self.items[image_id]

    def signature(self, item):
        if digest(item['fcd_path']) != item['fcd_hash']:
            raise Conflict('The Facad source changed. Restart the workbench to re-index it.')
        with Image.open(item['image_path']) as image:
            if image.size != (item['width'], item['height']):
                raise Conflict('Image dimensions changed. Restart the workbench to re-index it.')
        return hashlib.sha256((item['fcd_hash'] + digest(item['image_path'])).encode()).hexdigest()

    def catalog(self):
        saved = self.store.all()
        items = []
        for item in self.items.values():
            review = saved.get(item['id'], {})
            items.append({'id': item['id'], 'serial': item['serial'], 'image_index': item['image_index'],
                          'tracing_count': len(item['tracings']), 'status': review.get('status', 'pending'),
                          'revision': review.get('revision', 0),
                          'warning': any(t['out_of_bounds'] or t['warnings'] for t in item['tracings'])})
        return {'items': items, 'issues': self.issues, 'landmarks': self.landmarks,
                'demo': self.demo, 'dataset_id': self.dataset_id}

    def detail(self, image_id):
        item = self.item(image_id)
        public = {key: value for key, value in item.items()
                  if key not in {'image_path', 'fcd_path', 'fcd_hash', 'scales', 'image_count'}}
        signature = self.signature(item)
        review = self.store.get(image_id)
        return dict(public, source_signature=signature, review=review,
                    stale=bool(review and review['source_signature'] != signature),
                    imported_selection=self.imported.get((item['serial'], item['image_index'])),
                    history=[{key: row[key] for key in ('revision', 'status', 'reviewer', 'saved_at')}
                             for row in self.store.history(image_id)])

    def save(self, image_id, payload):
        item = self.item(image_id)
        if not isinstance(payload, dict):
            raise ReviewError('Expected a review object.')
        signature = self.signature(item)
        if payload.get('source_signature') != signature:
            raise Conflict('The source image changed. Reload it before saving.')
        status = payload.get('status')
        if status not in {'draft', 'accepted', 'excluded'}:
            raise ReviewError('Choose draft, accepted, or excluded.')
        reviewer = payload.get('reviewer', '')
        notes = payload.get('notes', '')
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
            raise ReviewError('Enter a reviewer name (up to 200 characters).')
        if not isinstance(notes, str) or len(notes) > 10000:
            raise ReviewError('Notes must be text, up to 10,000 characters.')
        revision = payload.get('revision')
        if type(revision) is not int or revision < 0:
            raise ReviewError('Invalid review revision.')
        tracing = payload.get('tracing_number')
        if type(tracing) is not int or not 1 <= tracing <= len(item['tracings']):
            if status != 'excluded' or tracing is not None:
                raise ReviewError('Select a valid source tracing.')
        points = payload.get('points')
        if not isinstance(points, dict) or len(points) > 1000:
            raise ReviewError('Invalid landmark points.')
        allowed = set(self.landmarks)
        for source in item['tracings']:
            allowed.update(source['marker_names'])
        clean = {}
        for name, point in points.items():
            if name not in allowed or not isinstance(point, list) or len(point) != 2:
                raise ReviewError(f'Invalid landmark: {name}')
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ReviewError(f'{name}: coordinates must be finite numbers.')
            clean[name] = [float(v) for v in point]
        if status == 'accepted' and (not clean or self.outside(clean, item['width'], item['height'])):
            raise ReviewError('Accepted tracings need at least one point and all points inside the image. '
                              'Correct or mark unavailable points missing, or exclude this image.')
        if status == 'accepted':
            try:
                with Image.open(item['image_path']) as image:
                    image.load()
            except (OSError, ValueError) as error:
                raise ReviewError('The original image cannot be decoded. Repair the source or exclude it.') from error
        return self.store.save(image_id, revision, {
            'imagefile': image_id, 'serial': item['serial'], 'image_index': item['image_index'],
            'status': status, 'reviewer': reviewer.strip(), 'notes': notes,
            'tracing_number': tracing, 'points': clean, 'source_signature': signature,
            'pixel_spacing_mm': item['pixel_spacing_mm'],
            'original_points': item['tracings'][tracing - 1]['points'] if tracing else {},
        })

    def preview(self, image_id):
        """Display PNG, same native dimensions; never use it as a training image."""
        item = self.item(image_id)
        with Image.open(item['image_path']) as source:
            if source.mode.startswith('I') or source.mode == 'F':
                array = np.asarray(source, dtype=np.float64)
                finite = array[np.isfinite(array)]
                low, high = np.percentile(finite, [0.5, 99.5]) if finite.size else (0, 1)
                array = np.nan_to_num(array, nan=low, posinf=high, neginf=low)
                array = np.clip((array - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)
                image = Image.fromarray(array)
            else:
                image = source.convert('RGB')
            buffer = io.BytesIO()
            image.save(buffer, 'PNG')
        return buffer.getvalue()

    def export(self):
        """Snapshot accepted records into a loader-compatible ZIP, with audit history."""
        reviews = self.store.all()
        accepted = {key: row for key, row in reviews.items()
                    if row['status'] == 'accepted' and key in self.items}
        if not accepted:
            raise ReviewError('Accept at least one image before exporting a training round.')
        for image_id, review in accepted.items():
            if review['source_signature'] != self.signature(self.item(image_id)):
                raise Conflict('An accepted image has changed since review. Re-review it before export.')
        # Keep temporary patient-containing archives within the chosen private workspace.
        archive = tempfile.TemporaryFile(dir=self.output_dir)
        try:
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
                groups = {}
                for image_id, review in accepted.items():
                    item = self.item(image_id)
                    groups.setdefault(item['serial'], []).append((item, review))
                    output.write(item['image_path'], f'images/{image_id}')
                    # Recheck after copy so a concurrently edited source cannot silently enter a round.
                    if review['source_signature'] != self.signature(item):
                        raise Conflict('An image changed during export. Retry after re-review.')
                for serial, entries in groups.items():
                    root = ET.Element('facad')
                    patient = ET.SubElement(root, 'patient')
                    template = entries[0][0]
                    for index in range(1, template['image_count'] + 1):
                        node = ET.SubElement(patient, 'imagefile')
                        ET.SubElement(node, 'idnr').text = f'curation-image-{index}'
                        matching = next((item for item, _ in entries if item['image_index'] == index), None)
                        ET.SubElement(node, 'name').text = matching['id'] if matching else ''
                        ET.SubElement(node, 'scale').text = template['scales'][index - 1]
                    for item, review in entries:
                        case = ET.SubElement(patient, 'case')
                        ET.SubElement(case, 'xray_ref').text = f"curation-image-{item['image_index']}"
                        ET.SubElement(case, 'name').text = 'Curated training reference'
                        for name, (x, y) in sorted(review['points'].items()):
                            marker = ET.SubElement(case, 'marker')
                            ET.SubElement(marker, 'name').text = name
                            ET.SubElement(marker, 'coord').text = f'{x:.8f} {y:.8f}'
                    # facad_io parses Latin-1; character references keep non-ASCII names lossless.
                    output.writestr(f'data/{serial}.fcd', ET.tostring(root, encoding='ascii'))
                selection = io.StringIO(newline='')
                writer = csv.writer(selection, delimiter=';', lineterminator='\n')
                for item in self.items.values():
                    writer.writerow([f"{item['serial']}_{item['image_index']}_review.png",
                                     '1' if item['id'] in accepted else 'x', ''])
                output.writestr('review_selection.csv', selection.getvalue())
                names = sorted({name for row in accepted.values() for name in row['points']})
                table = io.StringIO(newline='')
                fields = ['imagefile', 'serial', 'image_index', 'source_tracing', 'reviewer',
                          'revision', 'pixel_spacing_mm'] + [f'{name}_{axis}' for name in names for axis in ('x', 'y')]
                writer = csv.DictWriter(table, fields, lineterminator='\n')
                writer.writeheader()
                for image_id, review in sorted(accepted.items()):
                    row = {key: review[key] for key in fields[:7] if key in review}
                    row['source_tracing'] = review['tracing_number']
                    for name, (x, y) in review['points'].items():
                        row[f'{name}_x'], row[f'{name}_y'] = x, y
                    writer.writerow(row)
                output.writestr('cleaned_landmarks.csv', table.getvalue())
                output.writestr('audit.json', encode_json({
                    'schema_version': 1, 'dataset_id': self.dataset_id,
                    'exported_at': datetime.now(timezone.utc).isoformat(), 'synthetic_demo': self.demo,
                    'reviews': reviews,
                    'history': {key: [row for row in self.store.history(key)
                                      if row['revision'] <= review['revision']]
                                for key, review in reviews.items()},
                }))
                output.writestr('README.txt',
                    'OpenCeph curated training round\n\n'
                    'Only accepted images are included. Original image bytes and native coordinates are preserved.\n'
                    'data/*.fcd is a minimal training interchange format for facad_io, not a Facad clinical re-import.\n'
                    'Use images/, data/, and review_selection.csv with CephalometricDataset.\n'
                    'Selection 1 refers to the new curated case; audit.json records the original source tracing.\n'
                    'Keep this archive private: it contains source images, review metadata, and identifiers.\n'
                    'Freeze a new patient-level fold manifest for a new cohort; do not bypass existing split checks.\n')
            archive.seek(0)
            return archive
        except Exception:
            archive.close()
            raise


def create_demo(directory):
    """Procedural diagram fixtures: deliberately synthetic, never patient data."""
    images, data = Path(directory) / 'images', Path(directory) / 'data'
    images.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    base = {'S': [320, 235], 'N': [558, 205], 'Or': [543, 318], 'Po': [287, 333],
            'A': [580, 432], 'B': [563, 533], 'Sp': [601, 399], 'Pg': [593, 592],
            'Gn': [556, 637], 'Pt': [419, 337], 'iGo': [344, 570], 'pGo': [300, 528],
            'Is': [585, 468], 'Isa': [548, 414], 'Ii': [570, 480], 'Iia': [532, 537]}
    for example in range(1, 4):
        serial = f'd900{example}'
        image = Image.new('L', (800, 800), 18)
        draw = ImageDraw.Draw(image)
        draw.ellipse((153, 72, 584, 481), outline=115, width=12)
        draw.ellipse((169, 87, 571, 469), outline=54, width=3)
        draw.line([(544, 137), (558, 205), (574, 256), (624, 340), (584, 368),
                   (609, 399), (587, 437), (612, 477), (592, 530), (624, 581),
                   (593, 621), (554, 652), (349, 600), (282, 530), (285, 344)], fill=165, width=7)
        draw.line([(286, 345), (322, 488), (344, 569), (556, 637), (593, 592)], fill=104, width=9)
        draw.ellipse((479, 249, 567, 326), outline=154, width=5)
        draw.line([(417, 337), (502, 382), (601, 399)], fill=107, width=5)
        for x in range(454, 588, 21):
            draw.line([(x, 434), (x + 17, 469), (x + 11, 499)], fill=135, width=9)
        draw.line([(548, 414), (585, 468), (570, 480), (532, 537)], fill=191, width=6)
        draw.ellipse((304, 222, 336, 246), outline=180, width=4)
        draw.text((30, 30), 'SYNTHETIC DIAGRAM - NOT A PATIENT RADIOGRAPH', fill=200)
        image.save(images / f'{serial}_1.png')
        root = ET.Element('facad')
        patient = ET.SubElement(root, 'patient')
        node = ET.SubElement(patient, 'imagefile')
        ET.SubElement(node, 'idnr').text = '1'
        ET.SubElement(node, 'scale').text = '0.1'
        for number in range(1, 4):
            case = ET.SubElement(patient, 'case')
            ET.SubElement(case, 'xray_ref').text = '1'
            ET.SubElement(case, 'author').text = f'Demo reviewer {number}'
            for offset, (name, (x, y)) in enumerate(base.items()):
                if number == 3 and name in {'Isa', 'Iia'}:
                    continue
                marker = ET.SubElement(case, 'marker')
                ET.SubElement(marker, 'name').text = name
                delta = (number - 1) * (8 if offset % 3 == 0 else -5) + example - 1
                ET.SubElement(marker, 'coord').text = f'{x + delta} {y - delta / 2}'
        ET.ElementTree(root).write(data / f'{serial}.fcd', encoding='ascii')
    return images, data


def clean_path(text):
    """Accept pasted paths, including Windows 'Copy as path' quoting and ~."""
    if not isinstance(text, str):
        raise ReviewError('Expected a folder path.')
    text = text.strip().strip('"\'').strip()
    return Path(text).expanduser() if text else None


def folder_roots():
    roots = [Path.home()]
    if sys.platform == 'win32':
        roots += [Path(f'{letter}:\\') for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ'
                  if os.path.isdir(f'{letter}:\\')]
    else:
        roots.append(Path('/'))
    return [str(root) for root in roots]


def browse_folder(text):
    """List sub-folders of one directory so the page can offer a folder picker.

    Only directory names and file counts leave this function, never file contents.
    """
    path = clean_path(text) or Path.home()
    try:
        path = path.resolve()
        if not path.is_dir():
            raise ReviewError('That path is not a folder.')
        folders, images, facad = [], 0, 0
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        if not entry.name.startswith(('.', '$')):
                            folders.append(entry.name)
                    elif entry.is_file():
                        suffix = os.path.splitext(entry.name)[1].lower()
                        images += suffix in IMAGE_EXTENSIONS
                        facad += suffix == '.fcd'
                except OSError:
                    continue
    except PermissionError as error:
        raise ReviewError('Permission denied for that folder.') from error
    except OSError as error:
        raise ReviewError(f'Cannot read that folder: {error.strerror or error}') from error
    folders.sort(key=str.lower)
    return {'path': str(path), 'parent': str(path.parent) if path.parent != path else None,
            'folders': folders[:3000], 'truncated': len(folders) > 3000,
            'image_files': images, 'facad_files': facad, 'roots': folder_roots()}


def make_handler(workbench, token, host_header, defaults=None):
    """`workbench` may be None: the page then asks the user to choose folders."""
    defaults = defaults or {}
    current = {'workbench': workbench}

    def open_dataset(payload):
        if not isinstance(payload, dict):
            raise ReviewError('Expected folder choices.')
        images, facad = clean_path(payload.get('images')), clean_path(payload.get('facad'))
        if images is None or facad is None:
            raise ReviewError('Choose both the images folder and the Facad (.fcd) folder.')
        output = clean_path(payload.get('output')) or defaults.get('output') or Path('outputs/curation')
        try:
            opened = Workbench(images, facad, output, defaults.get('landmarks'),
                               defaults.get('review_csv'))
        except OSError as error:
            raise ReviewError(f'Cannot use those folders: {error.strerror or error}') from error
        current['workbench'] = opened
        return opened

    def catalog():
        active = current['workbench']
        folders = {'output': str(Path(defaults.get('output') or 'outputs/curation').resolve())}
        if active is None:
            return dict({'items': [], 'issues': [], 'landmarks': {}, 'demo': False,
                         'dataset_id': None}, loaded=False, folders=folders, token=token)
        folders.update(images=str(active.images_dir), facad=str(active.fcd_dir),
                       output=str(active.output_dir))
        return dict(active.catalog(), loaded=True, folders=folders, token=token)

    def require_workbench():
        if current['workbench'] is None:
            raise ReviewError('Open a dataset first.')
        return current['workbench']

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Do not print patient filenames in access logs.
            pass

        def send(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = encode_json(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                             "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if self.headers.get('Host') != host_header:
                self.send(403, {'error': 'Open the localhost URL printed by the server.'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{host_header}':
                self.send(403, {'error': 'Cross-origin requests are disabled.'})
                return False
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send(403, {'error': 'Cross-site requests are disabled.'})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            path = urlparse(self.path).path
            try:
                from urllib.parse import parse_qs
                query = parse_qs(urlparse(self.path).query)
                image_id = query.get('id', [''])[0]
                if path == '/api/catalog':
                    self.send(200, catalog())
                elif path == '/api/detail':
                    self.send(200, require_workbench().detail(image_id))
                elif path == '/api/image':
                    self.send(200, require_workbench().preview(image_id), 'image/png')
                elif path in {'/', '/app.js', '/style.css'}:
                    filename = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
                    content_type = {'/': 'text/html; charset=utf-8', '/app.js': 'text/javascript; charset=utf-8',
                                    '/style.css': 'text/css; charset=utf-8'}[path]
                    self.send(200, (STATIC / filename).read_bytes(), content_type)
                else:
                    self.send(404, {'error': 'Not found.'})
            except Conflict as error:
                self.send(409, {'error': str(error)})
            except ReviewError as error:
                self.send(400, {'error': str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.send(500, {'error': 'Could not read the requested data. Check source files and permissions.'})

        def do_POST(self):
            if not self.authorized():
                return
            if not secrets.compare_digest(self.headers.get('X-Review-Token', '').encode('utf-8'), token.encode('utf-8')):
                self.send(403, {'error': 'Reload the workbench to renew your session.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or length > 1024 * 1024:
                    raise ReviewError('Request is too large.')
                body = self.rfile.read(length)
                path = urlparse(self.path).path
                if path == '/api/browse':
                    payload = json.loads(body or b'{}')
                    self.send(200, browse_folder(payload.get('path', '') if isinstance(payload, dict) else ''))
                elif path == '/api/open':
                    open_dataset(json.loads(body or b'{}'))
                    self.send(200, catalog())
                elif path == '/api/export':
                    with require_workbench().export() as archive:
                        archive.seek(0, 2)
                        size = archive.tell()
                        archive.seek(0)
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/zip')
                        self.send_header('Content-Length', str(size))
                        self.send_header('Cache-Control', 'no-store')
                        self.send_header('Content-Disposition', 'attachment; filename="curated-training-round.zip"')
                        self.end_headers()
                        import shutil
                        shutil.copyfileobj(archive, self.wfile)
                elif path == '/api/review':
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise ReviewError('Expected a review object.')
                    self.send(200, require_workbench().save(payload.get('image_id'), payload))
                else:
                    self.send(404, {'error': 'Not found.'})
            except Conflict as error:
                self.send(409, {'error': str(error)})
            except (ReviewError, json.JSONDecodeError, TypeError, ValueError) as error:
                self.send(400, {'error': str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.send(500, {'error': 'Save or export failed. Check output permissions and available disk space.'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, help='Folder of <serial>_<image_index> images '
                        '(optional: omit it and choose folders in the browser)')
    parser.add_argument('--facad', type=Path, help='Folder of <serial>.fcd files')
    parser.add_argument('--output', type=Path, default=Path('outputs/curation'))
    parser.add_argument('--landmarks', type=Path, help='Optional landmark vocabulary YAML')
    parser.add_argument('--review-csv', type=Path, help='Existing review choices, offered as suggestions only')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--demo', action='store_true', help='Use generated synthetic diagrams')
    parser.add_argument('--open', action='store_true', help='Open the workbench in your default browser')
    args = parser.parse_args()
    if args.demo and (args.images or args.facad):
        parser.error('--demo cannot be combined with real input folders.')
    if args.demo:
        args.images, args.facad = create_demo(args.output / 'synthetic-inputs')
    elif bool(args.images) != bool(args.facad):
        parser.error('Supply both --images and --facad, or neither to choose folders in the browser.')
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    try:
        workbench = None
        if args.images:
            workbench = Workbench(args.images, args.facad, args.output, args.landmarks, args.review_csv, args.demo)
        landmarks = args.landmarks
        if landmarks is None:
            # Offer the repository's active vocabulary when folders are chosen in the page.
            bundled = Path(__file__).resolve().parents[2] / 'configs' / 'landmarks_15.yaml'
            landmarks = bundled if bundled.is_file() else None
        defaults = {'output': args.output.resolve(), 'landmarks': landmarks, 'review_csv': args.review_csv}
        host_header = f'127.0.0.1:{args.port}'
        server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                     make_handler(workbench, secrets.token_hex(32), host_header, defaults))
    except (ReviewError, OSError) as error:
        parser.error(str(error))
    server.daemon_threads = True
    url = f'http://{host_header}'
    print(f'OpenCeph review workbench: {url}')
    if workbench:
        print(f'{len(workbench.items)} images indexed; {len(workbench.issues)} import issues. '
              'Review storage: chosen output directory / reviews.sqlite3. Ctrl+C to stop.')
    else:
        print('No dataset loaded: choose the images and Facad folders in the browser. Ctrl+C to stop.')
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
