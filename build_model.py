"""Convert BodyParts3D IS-A OBJ files into compact per-system binary bundles + manifest."""
import os, re, sys, json, collections
import numpy as np
import pyfqmr

HERE = os.path.dirname(os.path.abspath(__file__))
BP = os.environ.get('BP3D_DIR', os.path.join(HERE, 'bp3d'))
OBJ_DIR = os.path.join(BP, 'isa_BP3D_4.0_obj_99')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out')
RATIO = float(os.environ.get('RATIO', '0.2'))
os.makedirs(OUT, exist_ok=True)

# ---- concept containment (FJ -> names of every concept that contains it) ----
containers = collections.defaultdict(set)
concept_size = collections.Counter()
concept_name = {}
for fn in ('isa_element_parts.txt', 'partof_element_parts.txt'):
    for line in open(os.path.join(BP, fn)).read().splitlines()[1:]:
        cid, name, fj = line.split('\t')
        containers[fj].add(name.lower())
        concept_size[(fn, cid)] += 1
        concept_name[(fn, cid)] = name
# most specific isa concept per FJ (for unnamed files)
fj_specific = {}
for line in open(os.path.join(BP, 'isa_element_parts.txt')).read().splitlines()[1:]:
    cid, name, fj = line.split('\t')
    k = ('isa_element_parts.txt', cid)
    if fj not in fj_specific or concept_size[k] < concept_size[fj_specific[fj]]:
        fj_specific[fj] = k


def has(N, *words):
    return any(w in N for w in words)


NAME_RULES = [
    ('skin', r'^skin$|hair|eyebrow|external ear|^lip$'),
    ('heart', r'papillary muscle|atrium|atrial|ventricle of heart|cavity of (left|right) (atrium|ventricle)|wall of (left |right )?(atrium|ventricle)|^wall of ventricle$|valve|cusp|chordae|interventricular septum|interatrial|myocard|trabecula|auricle|\bheart\b'),
    ('nervous', r'cerebr|cerebell|brain|gyrus|thalam|putamen|pallidus|caudate nucleus|amygdal|hippocamp|fornix|commissure|corpus callosum|collicul|geniculate|optic|chiasm|mammillary|midbrain|\bpons\b|medulla oblongata|spinal cord|(lateral|third|fourth) ventricle|choroid plexus|septum of telencephalon|lamina terminalis|stria |stria$|tentorium|falx|interpeduncular|interventricular foramen|nerve|ganglion|lobe of (left |right )?cerebral|occipital lobe|frontal lobe|parietal lobe|temporal lobe|insula|white matter|grey matter|gray matter|cauda equina|aqueduct|vermis|hypothalam|pineal'),
    ('nervous', r'eyeball|cornea|\blens\b|retina|corona ciliaris|ciliary|\biris\b|sclera|vitreous'),
    ('digestive', r'\bliver\b|hepatovenous|hepatic duct|biliary|bile|cystic duct|gallbladder'),
    ('veins', r'vein|venous|vena|\bsinus\b'),
    ('arteries', r'arter|aorta|palmar arch|plantar arch|arterial|celiac trunk|costocervical trunk|thyrocervical trunk|pulmonary trunk|brachiocephalic trunk'),
    ('respiratory', r'bronch|trachea|lung|laryn|epiglott|conus elasticus|arytenoid|cricoid|corniculate|cuneiform cartilage|thyroid cartilage|cricothyroid ligament|thyrohyoid membrane|nasal cartilage|alar cartilage|vocal'),
    ('muscles', r'muscle|tendon|trapezius|interosse(i|ous muscle)|lumbrical|levator|intertransversari|interspinal|aponeuro|iliotibial|linea alba|tendinous ring|raphe|diaphragm|platysma|fascia'),
    ('digestive', r'stomach|intestin|colon|duoden|jejun|ileum|ileocecal|cecum|append|rectum|esophag|pancrea|tongue|gingiva|pharyn|salivary|submandibular|sublingual|parotid|mesent|mesocolon|taenia|palate|uvula|omentum'),
    ('urinary', r'kidney|ureter|bladder|urethra'),
    ('endocrine', r'adrenal|suprarenal|thyroid gland|lobe of thyroid|pituitary|hypophysis|parathyroid'),
    ('reproductive', r'testis|epididym|seminal|prostate|penis|glans|cavernosum|spongiosum|deferent|ductus deferens|scrot'),
    ('lymphatic', r'spleen|thymus|lymph|tonsil'),
]


def classify(name, N):
    n = name.lower()
    for sysname, pat in NAME_RULES:
        if re.search(pat, n):
            return sysname
    if has(N, 'muscle organ', 'zone of muscle organ', 'head of muscle organ', 'tendon'):
        return 'muscles'
    if has(N, 'bone organ', 'cartilage organ', 'tooth', 'ligament organ', 'skeletal ligament', 'intervertebral disk',
           'articular disk of symphysis', 'skeletal system', 'ligament organ component') or re.search(r'bone|ligament|cartilage|disk|disc|membrane|menisc|tooth|incisor|canine|molar|vertebra|rib\b|sternum|skull', n):
        return 'skeleton'
    if 'vein' in N: return 'veins'
    if 'artery' in N: return 'arteries'
    if 'heart' in N: return 'heart'
    if has(N, 'nervous system', 'nerve', 'neuraxis', 'brain', 'left eye', 'right eye'): return 'nervous'
    if 'respiratory system' in N: return 'respiratory'
    if has(N, 'alimentary system', 'pancreaticobiliary system', 'gastrointestinal tract', 'liver', 'mouth'): return 'digestive'
    if 'urinary system' in N: return 'urinary'
    if 'endocrine system' in N: return 'endocrine'
    if 'cardiovascular system' in N: return 'heart'
    return 'other'


REGION_RULES = [
    ('Upper limb', r'upper limb|\barm\b|forearm|hand|wrist|finger|thumb|carp|metacarp|humer|radius|ulna|brachi|axill|palm|digit of hand|pollicis|deltoid|biceps brachii|triceps|scapula|clavicle|pectoral girdle|shoulder|interosse.*hand|lumbrical.*hand'),
    ('Lower limb', r'lower limb|\bleg\b|thigh|foot|toe|ankle|tars|metatars|femur|femor|tibia|fibula|patella|knee|gluteal|glute|popliteal|plantar|hallucis|sartorius|gracilis|quadriceps|hamstring|calcane|talus|pelvic girdle|hip bone|ilium|ischium|pubis|saphenous|peroneal|fibular'),
    ('Head', r'\bhead\b|brain|cerebr|cerebell|skull|cranial|cranium|face|facial|eye|orbit|nasal|nose|mouth|tooth|tongue|mandible|maxilla|temporal|frontal|parietal|occipital|sphenoid|ethmoid|zygom|palat|lingual|ophthalmic|trigeminal|mastica|ear\b|gyrus|lobe of brain|ventric.*brain|hippocamp|thalam|pons|medulla oblongata|midbrain|basilar'),
    ('Neck', r'neck|cervical|carotid|thyroid|larynx|pharynx|hyoid|jugular|vertebral artery|sternocleidomastoid|scalene|trachea'),
    ('Thorax', r'thora|chest|heart|lung|bronch|pulmonary|rib\b|ribs|sternum|costal|intercostal|mediastin|esophag|aorta|coronary|cardiac|atri|ventricle of heart|vena cava|azygos|diaphragm|breast|pericard'),
    ('Abdomen', r'abdom|liver|stomach|intestin|colon|duoden|jejun|ileum|pancrea|spleen|kidney|renal|adrenal|suprarenal|gallbladder|bile|hepatic|mesenteric|celiac|portal|lumbar|cecum|appendix|gastric|splenic|ureter'),
    ('Pelvis', r'pelvi|bladder|rectum|prostate|sacr|coccy|iliac|testis|penis|seminal|scrot|anal|perine'),
    ('Back', r'vertebra|vertebral column|spinal|back|postvertebral|erector|trapezius|latissimus|rhomboid|spinalis|longissimus|iliocostalis|multifidus'),
]


def region_of(name, N, cy):
    text = name.lower() + ' | ' + ' | '.join(sorted(N))
    own = name.lower()
    for reg, pat in REGION_RULES:  # own name first
        if re.search(pat, own):
            return reg
    for reg, pat in REGION_RULES:
        if re.search(pat, text):
            return reg
    # fall back to height (mm, original z)
    if cy > 1480: return 'Head'
    if cy > 1380: return 'Neck'
    if cy > 1120: return 'Thorax'
    if cy > 950: return 'Abdomen'
    if cy > 820: return 'Pelvis'
    return 'Lower limb'


def read_obj(path):
    head = {}
    vs, fs = [], []
    with open(path) as fh:
        for line in fh:
            c = line[:2]
            if c == 'v ':
                vs.append(line[2:])
            elif c == 'f ':
                fs.append(line[2:].replace('//', ' ').split()[0::2])
            elif line.startswith('# '):
                m = re.match(r'# ([^:]+?) : (.*)', line)
                if m: head[m.group(1).strip()] = m.group(2).strip()
    V = np.array([l.split() for l in vs], dtype=np.float64)
    F = np.array(fs, dtype=np.int64) - 1
    return head, V, F


# ---- gather & group by concept ----
groups = collections.OrderedDict()
files = sorted(f for f in os.listdir(OBJ_DIR) if f.endswith('.obj'))
print('reading', len(files), 'files', flush=True)
for i, f in enumerate(files):
    fj = f[:-4]
    head, V, F = read_obj(os.path.join(OBJ_DIR, f))
    cid = head.get('Concept ID') or ''
    name = head.get('English name') or ''
    if not cid:
        k = fj_specific.get(fj)
        cid, name = (k[1], concept_name[k]) if k else (fj, fj)
    g = groups.setdefault(cid, {'fma': cid, 'name': name, 'fjs': [], 'V': [], 'F': [], 'N': set()})
    g['fjs'].append(fj)
    g['F'].append(F + sum(len(v) for v in g['V']))
    g['V'].append(V)
    g['N'] |= containers.get(fj, set())
    if i % 200 == 0: print(' ', i, flush=True)


# ---- smooth groin: no external genitals ----
# The penis, testes, epididymides and pubic hair are dropped, the cords that run down into the
# scrotum are trimmed, and the skin over the genitals is replaced by a smooth membrane.
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
DROP_RE = re.compile(r'penis|testis|epididym|pubic hair', re.I)
CORD_RE = re.compile(r'deferent duct|testicular (artery|vein)', re.I)
_g = []
for g in groups.values():
    if re.search(r'corpus cavernosum of penis|corpus spongiosum of penis|glans penis', g['name'], re.I):
        P = np.concatenate(g['V']); _g.append(P[P[:, 1] < -160])
    elif re.search(r'testis|epididym', g['name'], re.I):
        _g.append(np.concatenate(g['V']))
GENITAL_TREE = cKDTree(np.concatenate(_g))
for cid in [c for c, g in groups.items() if DROP_RE.search(g['name'])]:
    print('  removing', groups[cid]['name'])
    del groups[cid]


def in_scrotum(V):   # original coords (mm): below the pubis, in front of the perineum, between the thighs
    return (V[:, 2] < 785) & (V[:, 1] < -118) & (np.abs(V[:, 0]) < 48)


def smooth_groin(V, F, radius=16.0, grow=3):
    region = GENITAL_TREE.query(V, distance_upper_bound=radius)[0] < radius
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    for _ in range(grow):   # widen a little so the new surface blends into the thighs and belly
        region = region | (A @ region.astype(np.float64) > 0) & in_scrotum(V) | region
    I = np.where(region)[0]
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(deg) - A
    LII = L[I][:, I].tocsc()
    B = ~region
    rhs = -(L[I][:, B] @ V[B])
    Vn = V.copy()
    for k in range(3):
        Vn[I, k] = spla.spsolve(LII, rhs[:, k])
    print('  smoothed', len(I), 'skin vertices over the groin')
    return Vn


parts = []
total_in = total_out = 0
for cid, g in groups.items():
    V = np.concatenate(g['V']); F = np.concatenate(g['F'])
    total_in += len(F)
    name = g['name'][0].upper() + g['name'][1:]
    sysname = classify(name, g['N'])
    cy = float(V[:, 2].mean())
    region = region_of(name, g['N'], cy)
    if name == 'Skin':
        region = 'Whole body'
    # decimate
    if sysname == 'skin':
        target = 130000
    else:
        target = min(20000, max(min(len(F), 160), int(len(F) * RATIO)))
    # weld duplicate vertices so the simplifier sees a connected surface
    key = np.round(V * 100).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = V[first]; F = inv.reshape(-1)[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    if name == 'Skin':
        V = smooth_groin(V, F)
    elif CORD_RE.search(name):
        F = F[~in_scrotum(V)[F].any(axis=1)]
    elif name == 'Urethra':
        F = F[~(V[F][:, :, 1] < -155).any(axis=1)]
    V2, F2 = V, F
    if target < len(F):
        for pb in (True, False):
            s = pyfqmr.Simplify()
            s.setMesh(V, F)
            s.simplify_mesh(target_count=target, aggressiveness=7, preserve_border=pb, verbose=0)
            V2, F2, _ = s.getMesh()
            if len(F2) <= target * 1.3:
                break
    if len(F2) == 0:
        V2, F2 = V, F
    total_out += len(F2)
    # to Y-up, front = +Z :  (x, z, -y)
    P = np.stack([V2[:, 0], V2[:, 2], -V2[:, 1]], axis=1).astype(np.float64)
    side = 'L' if re.search(r'\bleft\b', name, re.I) else ('R' if re.search(r'\bright\b', name, re.I) else '')
    parts.append(dict(fma=cid, name=name, sys=sysname, region=region, side=side, fjs=g['fjs'], P=P, F=np.asarray(F2, dtype=np.int64)))

print('faces in', total_in, 'out', total_out)
# ---- write bundles ----
by_sys = collections.defaultdict(list)
for p in parts:
    by_sys[p['sys']].append(p)
manifest = {'source': 'BodyParts3D, (c) The Database Center for Life Science, CC BY 4.0', 'systems': {}, 'parts': []}
for sysname, plist in by_sys.items():
    buf = bytearray()
    for p in plist:
        P, F = p['P'], p['F']
        mn = P.min(0); mx = P.max(0)
        scale = np.maximum(mx - mn, 1e-6) / 65535.0
        q = np.round((P - mn) / scale).astype(np.int64) - 32768
        q = q.astype(np.int16)
        vOff = len(buf); buf += q.tobytes()
        while len(buf) % 4: buf += b'\0'
        i32 = len(P) > 65535
        iOff = len(buf); buf += F.astype(np.uint32 if i32 else np.uint16).tobytes()
        while len(buf) % 4: buf += b'\0'
        manifest['parts'].append(dict(
            id=p['fjs'][0], fma=p['fma'], name=p['name'], sys=sysname, region=p['region'], side=p['side'],
            vOff=vOff, vCount=len(P), iOff=iOff, iCount=int(F.size), i32=i32,
            min=[round(float(x), 3) for x in (mn + 32768 * scale)], scale=[float('%.6g' % s) for s in scale]))
    fn = 'sys_%s.b64.txt' % sysname   # base64 text: artifact hosting serves text, not raw binary
    import base64
    open(os.path.join(OUT, fn), 'w').write(base64.b64encode(bytes(buf)).decode())
    manifest['systems'][sysname] = {'file': fn, 'bytes': len(buf), 'count': len(plist)}
    print(sysname, len(plist), 'parts', '%.2f MB' % (len(buf) / 1e6))
json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), separators=(',', ':'))
print('manifest parts', len(manifest['parts']))
