# ============================================================
# GOOD CODE NO ERROR (angle + lines)  +  BOX IMAGES INTEGRATION
# ============================================================
# Same logic as the original "MODULE 6+7 — ROBUST MULTI-ANGLE
# LOCALIZATION (ORB ONLY)" cell, but now the right-hand "grid
# map" panel in Part A (Col 1 / Col 2) and Part B (Col 1 / Col 2)
# is rendered from the pre-drawn YELLOW-BOX images located at:
#
#   C:\Users\Danial Hameed\Desktop\Module\cells
#       cell_001_boxes.jpg  ...  cell_100_boxes.jpg
#       (also matches .jpeg and .png)
#
# Behavior:
#   * If a cell_XXX_boxes.* image exists for the best DB image,
#     it is used as the right-side display in place of images[idx].
#   * Landmark coordinates (px/py) and the H-projected polygon
#     are rescaled from DB-image dims -> box-image dims so the
#     green match lines and small landmark squares stay aligned
#     with the yellow boxes already burnt into the cell image.
#   * ORB / RANSAC / BoVW / GPS-grid logic is unchanged.
#
# Requires: images[], df, LAT_COL, LON_COL, KAPPA_COL,
#           valid_indices, plus cv2, numpy, math, plt, tqdm,
#           MiniBatchKMeans (already loaded before this cell).
# ============================================================
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


# ── Cells / box-images directory ──────────────────────────────
CELLS_DIR = r'C:\Users\Danial Hameed\Desktop\Module\cells'


def _box_image_path(idx_1based):
    """
    Resolve the cell_XXX_boxes.{jpg,jpeg,png} path for a 1-based
    image index. Returns the first existing path or None.
    """
    base = os.path.join(CELLS_DIR, f'cell_{idx_1based:03d}_boxes')
    for ext in ('.jpg', '.jpeg', '.png'):
        p = base + ext
        if os.path.exists(p):
            return p
    return None


# Cache so we do not hit disk repeatedly inside the plotting loops
_BOX_IMG_CACHE = {}


def load_box_image(idx_1based):
    """
    Load cell_XXX_boxes.* for the given 1-based image index.
    Returns a BGR numpy image, or None if not found / unreadable.
    """
    if idx_1based in _BOX_IMG_CACHE:
        return _BOX_IMG_CACHE[idx_1based]

    p = _box_image_path(idx_1based)
    if p is None:
        _BOX_IMG_CACHE[idx_1based] = None
        return None

    img = cv2.imread(p)
    if img is None or img.size == 0:
        _BOX_IMG_CACHE[idx_1based] = None
        return None

    _BOX_IMG_CACHE[idx_1based] = img
    return img


def get_display_db_image(db_idx_0based):
    """
    Return the image to USE FOR DISPLAY in the right-side panel
    for DB index db_idx_0based. Prefers the pre-drawn box image
    cell_{idx+1:03d}_boxes.*; falls back to images[db_idx_0based].
    """
    box = load_box_image(db_idx_0based + 1)
    if box is not None:
        return box
    return images[db_idx_0based]


# ============================================================
# MODULE 6+7 — ROBUST MULTI-ANGLE LOCALIZATION (ORB ONLY)
# ============================================================

def create_structure_mask(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(edges, kernel, iterations=2)


def extract_hybrid_orb(gray, n=2500):
    orb = cv2.ORB_create(nfeatures=n, scaleFactor=1.2,
                         nlevels=8, edgeThreshold=10)
    mask = create_structure_mask(gray)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    _, df_ = orb.detectAndCompute(gray, None)
    _, dm = orb.detectAndCompute(masked, None)
    if df_ is not None and len(df_) > 1500:
        df_ = df_[:1500]
    if dm is not None and len(dm) > 1500:
        dm = dm[:1500]
    if df_ is not None and dm is not None:
        c = np.vstack((df_, dm))
    elif df_ is not None:
        c = df_
    elif dm is not None:
        c = dm
    else:
        return np.empty((0, 32), dtype=np.float32)
    if len(c) > 2500:
        c = c[:2500]
    return c.astype(np.float32)


def desc_to_bovw(descriptors, kmeans, k):
    hist = np.zeros(k, dtype=np.float32)
    if descriptors is None or descriptors.size == 0:
        return hist
    words = kmeans.predict(descriptors)
    hist = np.bincount(words, minlength=k).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def suppress_nearby(kps, min_dist=14, max_keep=20):
    kept = []
    for kp in kps:
        if not any(math.sqrt((kp.pt[0] - k2.pt[0]) ** 2 +
                             (kp.pt[1] - k2.pt[1]) ** 2) < min_dist
                   for k2 in kept):
            kept.append(kp)
        if len(kept) >= max_keep:
            break
    return kept


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def color_shift(img, hue_shift=10, sat_scale=1.15, val_scale=0.85):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def transform_query(img, angle=15, hue_shift=10, sat_scale=1.15, val_scale=0.85):
    return color_shift(rotate_image(img, angle), hue_shift, sat_scale, val_scale)


# ── BoVW Database (ORB) ───────────────────────────────────────
print('=' * 65)
print('STEP 1: Building ORB BoVW Database')
print('=' * 65)
K_WORDS = 100
all_desc_list, per_image_desc, valid_db_idx = [], [], []

for idx in tqdm(valid_indices, desc='ORB extraction'):
    img = images[idx]
    if img is None:
        continue
    gray = cv2.cvtColor(cv2.resize(img, (800, 600)), cv2.COLOR_BGR2GRAY)
    desc = extract_hybrid_orb(gray)
    if desc.size == 0:
        continue
    per_image_desc.append(desc)
    all_desc_list.append(desc)
    valid_db_idx.append(idx)

all_desc_np = np.vstack(all_desc_list)
print(f'Total ORB descriptors: {all_desc_np.shape[0]}')
kmeans_bovw = MiniBatchKMeans(n_clusters=K_WORDS, random_state=42,
                              batch_size=2000, n_init='auto')
kmeans_bovw.fit(all_desc_np)
db_histograms = np.array(
    [desc_to_bovw(d, kmeans_bovw, K_WORDS) for d in per_image_desc],
    dtype=np.float32)
print(f'BoVW DB: {db_histograms.shape}  ✓')

# ── Box-image availability report ─────────────────────────────
_n_box = sum(1 for i in range(1, 101) if _box_image_path(i) is not None)
print(f'Box images detected in CELLS_DIR ({CELLS_DIR}): {_n_box}/100')

# ── GPS Spatial Grid ──────────────────────────────────────────
GRID_SIZE = 10
lats_arr = np.array([float(df.iloc[i][LAT_COL]) for i in valid_db_idx])
lons_arr = np.array([float(df.iloc[i][LON_COL]) for i in valid_db_idx])
lat_min, lat_max = lats_arr.min(), lats_arr.max()
lon_min, lon_max = lons_arr.min(), lons_arr.max()
lat_span = max(1e-9, lat_max - lat_min)
lon_span = max(1e-9, lon_max - lon_min)
spatial_cells = {}
for pos, idx in enumerate(valid_db_idx):
    r = int(np.clip(((lats_arr[pos] - lat_min) / lat_span) * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
    c = int(np.clip(((lons_arr[pos] - lon_min) / lon_span) * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
    spatial_cells.setdefault(f'{r}_{c}', []).append(pos)


def get_spatial_candidates(q_lat, q_lon, radius=2):
    r0 = int(np.clip(((q_lat - lat_min) / lat_span) * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
    c0 = int(np.clip(((q_lon - lon_min) / lon_span) * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
    allowed = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            for pos in spatial_cells.get(f'{r0+dr}_{c0+dc}', []):
                allowed.add(pos)
    return list(allowed) if allowed else list(range(len(valid_db_idx)))


# ── Landmark DB ───────────────────────────────────────────────
print('\nSTEP 2: Building Landmark DB')
orb_lm = cv2.ORB_create(nfeatures=300, scaleFactor=1.2, nlevels=6,
                        edgeThreshold=15, fastThreshold=25)
landmark_db = {}
for idx in range(len(images)):
    img = images[idx]
    if img is None:
        continue
    img_lat = float(df.iloc[idx][LAT_COL])
    img_lon = float(df.iloc[idx][LON_COL])
    img_h, img_w = img.shape[:2]
    gray_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    kps, des = orb_lm.detectAndCompute(gray_eq, None)
    if not kps or des is None:
        continue
    kps_filtered = suppress_nearby(sorted(kps, key=lambda k: k.response, reverse=True))
    kp_all = list(kps)
    kept_indices = []
    for kp_f in kps_filtered:
        for ki, kp_o in enumerate(kp_all):
            if abs(kp_o.pt[0] - kp_f.pt[0]) < 0.5 and abs(kp_o.pt[1] - kp_f.pt[1]) < 0.5:
                kept_indices.append(ki)
                break
    landmarks = []
    for lm_id, ki in enumerate(kept_indices):
        kp = kps[ki]
        d = des[ki]
        px, py = kp.pt
        if idx < len(df) - 1:
            dlat = abs(float(df.iloc[idx + 1][LAT_COL]) - img_lat)
            dlon = abs(float(df.iloc[idx + 1][LON_COL]) - img_lon)
        else:
            dlat, dlon = 0.0005, 0.0005
        landmarks.append({'id': lm_id + 1, 'kp': kp, 'des': d,
                          'lat': img_lat + (0.5 - py / img_h) * dlat * 3,
                          'lon': img_lon + (px / img_w - 0.5) * dlon * 3,
                          'px': px, 'py': py, 'response': kp.response,
                          # original DB image dims — needed to rescale
                          # px/py to box-image dims when displaying
                          'src_w': img_w, 'src_h': img_h})
    landmark_db[idx] = landmarks
print(f'Landmark DB: {len(landmark_db)} images  ✓')


# ── RANSAC match — ORB only ───────────────────────────────────

def _ransac_match_single(gray_q, filtered_pos, scores_cos):
    orb_v = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2,
                           nlevels=8, edgeThreshold=10)
    bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    kp_q, des_q = orb_v.detectAndCompute(gray_q, None)
    if des_q is None or len(kp_q) < 4:
        return None
    best = None
    best_score = -1
    for pos in filtered_pos:
        db_idx = valid_db_idx[pos]
        db_img = images[db_idx]
        if db_img is None:
            continue
        gray_db = cv2.cvtColor(cv2.resize(db_img, (800, 600)), cv2.COLOR_BGR2GRAY)
        kp_db, des_db = orb_v.detectAndCompute(gray_db, None)
        if des_db is None or len(kp_db) < 4:
            continue
        raw = bf_ham.knnMatch(des_q, des_db, k=2)
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            continue
        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_db[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue
        mask_b = mask.ravel().astype(bool)
        n_inliers = int(mask_b.sum())
        if n_inliers < 4:
            continue
        score = 0.7 * n_inliers + 0.3 * len(good)
        inlier_matches = [m for m, k in zip(good, mask_b) if k]
        if score > best_score:
            best_score = score
            best = {'db_idx': db_idx, 'bovw_score': float(scores_cos[pos]),
                    'n_inliers': n_inliers, 'n_outliers': len(good) - n_inliers,
                    'n_good': len(good), 'score': score,
                    'inlier_ratio': n_inliers / (len(good) + 1e-6),
                    'inlier_matches': inlier_matches,
                    'kp_q': kp_q, 'kp_db': kp_db,
                    'gray_q': gray_q, 'gray_db': gray_db, 'H': H}
    return best


def find_best_match_multiangle(query_img_bgr, q_lat, q_lon,
                               top_k_bovw=10, ransac_radius=2):
    TEST_ANGLES = [0, 15, 30, 45, -15, -30]
    global_best = None
    global_best_score = -1
    global_best_angle = 0
    for angle in TEST_ANGLES:
        rotated = rotate_image(cv2.resize(query_img_bgr, (800, 600)), angle)
        gray_q = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        desc_q = extract_hybrid_orb(gray_q)
        hist_q = desc_to_bovw(desc_q, kmeans_bovw, K_WORDS).reshape(1, -1)
        scores_cos = cos_sim(hist_q, db_histograms)[0]
        ranked_pos = np.argsort(scores_cos)[::-1][:top_k_bovw]
        spatial_pos = set(get_spatial_candidates(q_lat, q_lon, radius=ransac_radius))
        filtered_pos = [p for p in ranked_pos if p in spatial_pos] or list(ranked_pos)
        result = _ransac_match_single(gray_q, filtered_pos, scores_cos)
        if result and result['score'] > global_best_score:
            global_best_score = result['score']
            global_best = result
            global_best_angle = angle
            global_best['winning_angle'] = angle
            global_best['rotated_query_bgr'] = rotated
    if global_best:
        print(f'  Best angle: {global_best_angle}°  '
              f'img_{global_best["db_idx"]+1}  '
              f'inliers={global_best["n_inliers"]}  '
              f'score={global_best["score"]:.1f}')
    return global_best


# ── Simulate UAV frames ───────────────────────────────────────
TARGET_IMAGE_IDX = 75
START_IMAGE_IDX = 0
nav_path = list(range(START_IMAGE_IDX, TARGET_IMAGE_IDX + 1))
sim_frames = []
for k in range(len(nav_path) - 1):
    i, j = nav_path[k], nav_path[k + 1]
    img_a, img_b = images[i], images[j]
    if img_a is None or img_b is None:
        continue
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    sim_frames.append({'type': 'real', 'idx': i, 'image': images[i],
                       'lat': float(df.iloc[i][LAT_COL]), 'lon': float(df.iloc[i][LON_COL]),
                       'kappa': float(df.iloc[i][KAPPA_COL]) if KAPPA_COL else 0.0,
                       'label': f'img_{i+1}'})
    blended = cv2.addWeighted(cv2.resize(img_a, (w, h)), 0.5,
                              cv2.resize(img_b, (w, h)), 0.5, 0)
    sim_frames.append({'type': 'synthetic', 'idx': None, 'image': blended,
                       'lat': (float(df.iloc[i][LAT_COL]) + float(df.iloc[j][LAT_COL])) / 2,
                       'lon': (float(df.iloc[i][LON_COL]) + float(df.iloc[j][LON_COL])) / 2,
                       'kappa': float(df.iloc[i][KAPPA_COL]) if KAPPA_COL else 0.0,
                       'label': f'syn_{i+1}_{j+1}'})
sim_frames.append({'type': 'real', 'idx': TARGET_IMAGE_IDX,
                   'image': images[TARGET_IMAGE_IDX],
                   'lat': float(df.iloc[TARGET_IMAGE_IDX][LAT_COL]),
                   'lon': float(df.iloc[TARGET_IMAGE_IDX][LON_COL]),
                   'kappa': float(df.iloc[TARGET_IMAGE_IDX][KAPPA_COL]) if KAPPA_COL else 0.0,
                   'label': f'img_{TARGET_IMAGE_IDX+1}'})
print(f'Simulated frames: {len(sim_frames)}  ✓')


# ── Landmark match col — STRAIGHT LINES (uses BOX image) ─────
def draw_landmark_matches_col(ax2, q_img, match_img, best_idx,
                              landmark_db, orb_lm, DISP=300):
    """
    Right-side landmark-match panel.

    The "grid map" image is the pre-drawn yellow-box image
    cell_{best_idx+1:03d}_boxes.* when present; if not, the raw DB
    image is used as a fallback.

    Landmark px/py and the right-side landmark squares are scaled
    from the raw DB image dims to the displayed (box) image dims
    so the green match lines and small squares stay aligned with
    the yellow boxes already burnt into the cell image.
    """
    if match_img is None or best_idx not in landmark_db:
        ax2.axis('off')
        return
    lm_list = landmark_db[best_idx]
    orig_h, orig_w = q_img.shape[:2]

    # Image actually shown on the right (box image when available)
    disp_img = get_display_db_image(best_idx)
    if disp_img is None:
        disp_img = match_img
    disp_h, disp_w = disp_img.shape[:2]

    # Raw DB image dims (px/py space for landmarks)
    db_h, db_w = match_img.shape[:2]

    # BOTH images resized to same DISP×DISP for drawMatches
    q_small = cv2.resize(q_img, (DISP, DISP))
    m_small = cv2.resize(disp_img, (DISP, DISP))

    gray_q = cv2.cvtColor(q_img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray_q)
    kps_q, des_q = orb_lm.detectAndCompute(gray_eq, None)
    if des_q is None or len(kps_q) < 4:
        ax2.axis('off')
        return

    lm_des_arr = np.array([lm['des'] for lm in lm_list], dtype=np.uint8)
    bf_c = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf_c.match(des_q, lm_des_arr)
    except Exception:
        matches = []
    matches = sorted(matches, key=lambda x: x.distance)[:15]
    if not matches:
        ax2.axis('off')
        return

    # Keypoint scaling: query (orig_w,orig_h) -> DISP, landmarks
    # (db_w,db_h in raw DB) -> DISP. Because q_img and disp_img are
    # both resized to DISP×DISP, drawMatches sees a consistent
    # coordinate frame and draws straight lines.
    kps_q_sc = [cv2.KeyPoint(kp.pt[0] * DISP / orig_w,
                             kp.pt[1] * DISP / orig_h,
                             kp.size, kp.angle, kp.response, kp.octave)
                for kp in kps_q]
    lm_kps_sc = [cv2.KeyPoint(lm['px'] * DISP / db_w,
                              lm['py'] * DISP / db_h,
                              lm['kp'].size, lm['kp'].angle,
                              lm['response'], lm['kp'].octave)
                 for lm in lm_list]

    match_vis = cv2.drawMatches(
        q_small, kps_q_sc,
        m_small, lm_kps_sc,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(100, 100, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Small landmark labels on the right side. Coordinates in the
    # DISPLAYED (box) image are scaled the same way as kp_lm above.
    for lm in lm_list:
        sx = int(np.clip(lm['px'] * DISP / db_w + DISP, DISP + 2, 2 * DISP - 2))
        sy = int(np.clip(lm['py'] * DISP / db_h, 2, DISP - 2))
        resp = lm['response']
        col = (255, 60, 0) if resp > 25 else \
              (255, 200, 0) if resp > 12 else (60, 200, 60)
        cv2.rectangle(match_vis, (sx - 4, sy - 4), (sx + 4, sy + 4), col, 1)
        cv2.putText(match_vis, str(lm['id']), (sx - 3, sy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, col, 1, cv2.LINE_AA)

    label_right = 'GRID MAP (BOX)' if disp_img is not match_img else 'GRID MAP'
    cv2.putText(match_vis, 'QUERY', (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(match_vis, label_right, (DISP + 5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    ax2.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Query ↔ Landmark Matches (box image)\n'
                  f'{len(matches)} straight connections',
                  fontsize=7, fontweight='bold')
    ax2.axis('off')


# ============================================================
# PART A — BEST MATCH SUMMARY
# ============================================================
test_frame_indices = [0, 10, 20, 30, 40, 50, 60, 70, -1]
DISPLAY_ANGLE = 30

fig_rows = len(test_frame_indices)
fig, axes = plt.subplots(fig_rows, 3, figsize=(22, fig_rows * 4))
localization_results = []

print('\n' + '=' * 70)
print('PART A — BEST MATCH SUMMARY (ORB | 0°,±15°,±30°,45°)')
print('Right-side panel uses cell_XXX_boxes.* when available')
print('=' * 70)

for row_i, fi in enumerate(test_frame_indices):
    frame = sim_frames[fi]
    q_img = frame['image']
    q_lat, q_lon = frame['lat'], frame['lon']
    q_label, q_type = frame['label'], frame['type']
    if q_img is None:
        continue

    q_display = transform_query(cv2.resize(q_img, (400, 400)), angle=DISPLAY_ANGLE)
    print(f'\n[{q_label}] Testing angles: 0°,±15°,±30°,45°')
    best = find_best_match_multiangle(q_img, q_lat, q_lon,
                                      top_k_bovw=10, ransac_radius=2)
    if best is None:
        print('  → NO MATCH FOUND')
        for ax in axes[row_i]:
            ax.axis('off')
        continue

    best_idx = best['db_idx']
    n_inliers = best['n_inliers']
    ratio = best['inlier_ratio']
    score = best['score']
    win_angle = best['winning_angle']
    est_lat = float(df.iloc[best_idx][LAT_COL])
    est_lon = float(df.iloc[best_idx][LON_COL])
    gps_err = math.sqrt((q_lat - est_lat) ** 2 + (q_lon - est_lon) ** 2) * 111000
    match_img = images[best_idx]
    box_img = load_box_image(best_idx + 1)
    used_box = box_img is not None

    localization_results.append({
        'frame_label': q_label, 'frame_type': q_type,
        'true_lat': q_lat, 'true_lon': q_lon,
        'est_lat': est_lat, 'est_lon': est_lon,
        'best_match': best_idx, 'n_inliers': n_inliers,
        'inlier_ratio': ratio, 'score': score,
        'winning_angle': win_angle, 'gps_err_m': gps_err,
        'box_image_used': used_box})

    # Col 0: query shown at 30°
    axes[row_i, 0].imshow(cv2.cvtColor(q_display, cv2.COLOR_BGR2RGB))
    axes[row_i, 0].set_title(
        f'Query: {q_label} [{q_type}]\n'
        f'Lat={q_lat:.5f} Lon={q_lon:.5f}\n'
        f'[shown {DISPLAY_ANGLE}° | won {win_angle}°]',
        fontsize=7, fontweight='bold')
    axes[row_i, 0].axis('off')

    # Col 1: best match (BOX image when available) + landmark
    # squares + H-projected polygon. Coordinates are rescaled
    # from raw DB image dims to the displayed box image dims.
    if match_img is not None:
        disp_img = box_img if used_box else match_img
        disp_h, disp_w = disp_img.shape[:2]
        db_h, db_w = match_img.shape[:2]

        m_vis = cv2.cvtColor(cv2.resize(disp_img, (400, 400)),
                             cv2.COLOR_BGR2RGB).copy()
        # Landmarks: scale from raw DB image -> 400×400 display
        for lm in landmark_db.get(best_idx, []):
            sx = int(lm['px'] * 400 / db_w)
            sy = int(lm['py'] * 400 / db_h)
            col = (255, 60, 0) if lm['response'] > 25 else \
                  (255, 200, 0) if lm['response'] > 12 else (60, 200, 60)
            cv2.rectangle(m_vis, (sx - 4, sy - 4), (sx + 4, sy + 4), col, 1)
            cv2.putText(m_vis, str(lm['id']), (sx - 3, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, col, 1, cv2.LINE_AA)
        # H polygon: H maps query 800×600 -> DB 800×600. Scale to
        # 400×400 display. This is dim-independent because the
        # raw DB image was resized to 800×600 inside RANSAC.
        h_q, w_q = best['gray_q'].shape
        corners = np.float32([[0, 0], [0, h_q - 1], [w_q - 1, h_q - 1],
                              [w_q - 1, 0]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, best['H'])
        proj[:, 0, 0] *= 400 / 800
        proj[:, 0, 1] *= 400 / 600
        cv2.polylines(m_vis, [np.int32(proj)], True, (255, 255, 0), 2, cv2.LINE_AA)

        # Mark when box image was used
        cv2.putText(m_vis,
                    'BOX IMG' if used_box else 'RAW DB',
                    (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0) if used_box else (0, 200, 255),
                    1, cv2.LINE_AA)

        axes[row_i, 1].imshow(m_vis)
    tc = ('darkgreen' if gps_err < 300 else
          'darkorange' if gps_err < 1000 else 'darkred')
    axes[row_i, 1].set_title(
        f'✅ Best: img_{best_idx+1} [won {win_angle}°]'
        f'{" (box)" if used_box else ""}\n'
        f'In={n_inliers} Sc={score:.1f} GPS≈{gps_err:.0f}m',
        fontsize=7, fontweight='bold', color=tc)
    axes[row_i, 1].axis('off')

    # Col 2: straight landmark lines, right side = box image
    draw_landmark_matches_col(axes[row_i, 2], q_img, match_img,
                              best_idx, landmark_db, orb_lm)

plt.suptitle(
    'MODULE 6+7 PART A — ORB Multi-Angle Best Match (box images)\n'
    'Tests 0°,±15°,±30°,45° → ORB BoVW→GPS→RANSAC; '
    'right-side panels use cell_XXX_boxes.* when present',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('localization_multiangle.png', dpi=120, bbox_inches='tight')
plt.show()
print('PART A shown ✓')

print('\n' + '=' * 70)
print(f'{"Frame":<18}{"Match":<10}{"Angle":<8}{"Inliers":<10}'
      f'{"Ratio":<8}{"Score":<10}{"GPS_err":<11}{"Box?"}')
print('-' * 78)
for r in localization_results:
    flag = '✅' if r['gps_err_m'] < 300 else '⚠️' if r['gps_err_m'] < 1000 else '❌'
    print(f'{r["frame_label"]:<18}img_{r["best_match"]+1:<6}'
          f'{r["winning_angle"]:>+4}°   {r["n_inliers"]:<10}'
          f'{r["inlier_ratio"]:<8.2f}{r["score"]:<10.1f}'
          f'{r["gps_err_m"]:.0f}m {flag:<6}'
          f'{"yes" if r["box_image_used"] else "no"}')


# ============================================================
# PART B — PER-FRAME ANGLE BREAKDOWN
# ============================================================
TEST_ANGLES = [0, 15, 30, 45, -15, -30]
DISP = 200
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print('\n' + '=' * 70)
print('PART B — Per-Frame Angle Breakdown (ORB) — box images on right')
print('=' * 70)

computed = []
for fi in tqdm(test_frame_indices, desc='Computing all angles'):
    frame = sim_frames[fi]
    q_img = frame['image']
    q_lat, q_lon = frame['lat'], frame['lon']
    q_label, q_type = frame['label'], frame['type']
    if q_img is None:
        continue
    orig_h, orig_w = q_img.shape[:2]
    angle_results = []

    gray_q_orig = cv2.cvtColor(q_img, cv2.COLOR_BGR2GRAY)
    gray_q_eq = cv2.createCLAHE(clipLimit=3.0,
                                tileGridSize=(4, 4)).apply(gray_q_orig)
    kps_q_orig, des_q_orig = orb_lm.detectAndCompute(gray_q_eq, None)

    for angle in TEST_ANGLES:
        rotated_bgr = rotate_image(cv2.resize(q_img, (800, 600)), angle)
        gray_rot = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)

        desc_q = extract_hybrid_orb(gray_rot)
        hist_q = desc_to_bovw(desc_q, kmeans_bovw, K_WORDS).reshape(1, -1)
        scores_cos = cos_sim(hist_q, db_histograms)[0]
        ranked_pos = np.argsort(scores_cos)[::-1][:10]
        spatial_pos = set(get_spatial_candidates(q_lat, q_lon, radius=2))
        filtered_pos = ([p for p in ranked_pos if p in spatial_pos]
                        or list(ranked_pos))
        best = _ransac_match_single(gray_rot, filtered_pos, scores_cos)

        lm_matches = []
        lm_list = []
        kps_q_sc = []
        lm_kps_sc = []

        if best is not None and des_q_orig is not None and len(kps_q_orig) >= 4:
            best_idx = best['db_idx']
            match_img2 = images[best_idx]
            db_h2, db_w2 = match_img2.shape[:2]
            lm_list = landmark_db.get(best_idx, [])

            if len(lm_list) >= 2:
                lm_des_arr = np.array([lm['des'] for lm in lm_list],
                                      dtype=np.uint8)
                try:
                    lm_matches = bf_orb.match(des_q_orig, lm_des_arr)
                    lm_matches = sorted(lm_matches,
                                        key=lambda x: x.distance)[:15]
                except Exception:
                    lm_matches = []

                kps_q_sc = [cv2.KeyPoint(kp.pt[0] * DISP / orig_w,
                                         kp.pt[1] * DISP / orig_h,
                                         kp.size, kp.angle,
                                         kp.response, kp.octave)
                            for kp in kps_q_orig]
                lm_kps_sc = [cv2.KeyPoint(lm['px'] * DISP / db_w2,
                                          lm['py'] * DISP / db_h2,
                                          lm['kp'].size, lm['kp'].angle,
                                          lm['response'], lm['kp'].octave)
                             for lm in lm_list]

        angle_results.append({
            'angle': angle, 'rotated': rotated_bgr, 'best': best,
            'lm_list': lm_list, 'lm_matches': lm_matches,
            'kps_q_sc': kps_q_sc, 'lm_kps_sc': lm_kps_sc})

    computed.append({'label': q_label, 'type': q_type,
                     'lat': q_lat, 'lon': q_lon,
                     'q_img': q_img, 'orig_h': orig_h, 'orig_w': orig_w,
                     'angles': angle_results})

for fr in computed:
    q_label = fr['label']
    q_lat = fr['lat']
    q_lon = fr['lon']
    q_img = fr['q_img']
    n_angles = len(fr['angles'])

    fig, axes = plt.subplots(n_angles, 3, figsize=(18, n_angles * 3.2))
    fig.patch.set_facecolor('#111122')
    fig.suptitle(
        f'PART B — {q_label} [{fr["type"]}]  '
        f'Lat={q_lat:.5f} Lon={q_lon:.5f}\n'
        f'Each row = one rotation angle tested independently  '
        f'(grid map = box image when available)',
        fontsize=11, fontweight='bold', color='white', y=1.01)

    for ang_i, ar in enumerate(fr['angles']):
        angle = ar['angle']
        best = ar['best']
        lm_list = ar['lm_list']
        lm_matches = ar['lm_matches']
        kps_q_sc = ar['kps_q_sc']
        lm_kps_sc = ar['lm_kps_sc']

        ax0, ax1, ax2 = axes[ang_i, 0], axes[ang_i, 1], axes[ang_i, 2]
        for ax in [ax0, ax1, ax2]:
            ax.set_facecolor('#111122')

        ax0.imshow(cv2.cvtColor(
            cv2.resize(ar['rotated'], (DISP, DISP)), cv2.COLOR_BGR2RGB))
        ax0.set_title(f'Query @ {angle:+d}°',
                      fontsize=9, fontweight='bold', color='white')
        ax0.axis('off')

        if best is None:
            ax1.set_facecolor('#330000')
            ax1.text(0.5, 0.5, 'NO MATCH\nFOUND', ha='center', va='center',
                     color='red', fontsize=11, fontweight='bold',
                     transform=ax1.transAxes)
            ax1.set_title(f'{angle:+d}° → No match', fontsize=8, color='red')
            ax1.axis('off')
            ax2.axis('off')
            continue

        best_idx = best['db_idx']
        n_inliers = best['n_inliers']
        score = best['score']
        match_img2 = images[best_idx]
        db_h2, db_w2 = match_img2.shape[:2]
        box_img2 = load_box_image(best_idx + 1)
        used_box = box_img2 is not None
        disp_img2 = box_img2 if used_box else match_img2

        est_lat = float(df.iloc[best_idx][LAT_COL])
        est_lon = float(df.iloc[best_idx][LON_COL])
        gps_err = math.sqrt((q_lat - est_lat) ** 2 +
                            (q_lon - est_lon) ** 2) * 111000

        # Col 1: best match (BOX image when available) + landmarks + polygon
        m_vis = cv2.cvtColor(cv2.resize(disp_img2, (DISP, DISP)),
                             cv2.COLOR_BGR2RGB).copy()
        for lm in lm_list:
            sx = int(np.clip(lm['px'] * DISP / db_w2, 4, DISP - 4))
            sy = int(np.clip(lm['py'] * DISP / db_h2, 4, DISP - 4))
            col = (255, 60, 0) if lm['response'] > 25 else \
                  (255, 200, 0) if lm['response'] > 12 else (60, 200, 60)
            cv2.rectangle(m_vis, (sx - 4, sy - 4), (sx + 4, sy + 4), col, 1)
            cv2.putText(m_vis, str(lm['id']), (sx - 3, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, col, 1, cv2.LINE_AA)
        h_q, w_q = best['gray_q'].shape
        corners = np.float32([[0, 0], [0, h_q - 1], [w_q - 1, h_q - 1],
                              [w_q - 1, 0]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, best['H'])
        proj[:, 0, 0] *= DISP / 800
        proj[:, 0, 1] *= DISP / 600
        cv2.polylines(m_vis, [np.int32(proj)], True, (255, 255, 0), 2, cv2.LINE_AA)
        if used_box:
            cv2.putText(m_vis, 'BOX', (5, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        tc = ('darkgreen' if gps_err < 300 else
              'darkorange' if gps_err < 1000 else 'darkred')
        ax1.imshow(m_vis)
        ax1.set_title(f'img_{best_idx+1} [{angle:+d}°]'
                      f'{" (box)" if used_box else ""}\n'
                      f'In={n_inliers} Sc={score:.0f} GPS≈{gps_err:.0f}m',
                      fontsize=7, fontweight='bold', color=tc)
        ax1.axis('off')

        # Col 2: rotated query ↔ landmark straight lines, right = box img
        q_rot_small = cv2.resize(ar['rotated'], (DISP, DISP))
        m_small = cv2.resize(disp_img2, (DISP, DISP))

        if lm_matches and kps_q_sc and lm_kps_sc:
            match_vis = cv2.drawMatches(
                q_rot_small, kps_q_sc,
                m_small, lm_kps_sc,
                lm_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(100, 100, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            for lm in lm_list:
                sx = int(np.clip(lm['px'] * DISP / db_w2 + DISP, DISP + 2, 2 * DISP - 2))
                sy = int(np.clip(lm['py'] * DISP / db_h2, 2, DISP - 2))
                col = (255, 60, 0) if lm['response'] > 25 else \
                      (255, 200, 0) if lm['response'] > 12 else (60, 200, 60)
                cv2.rectangle(match_vis, (sx - 4, sy - 4), (sx + 4, sy + 4), col, 1)
                cv2.putText(match_vis, str(lm['id']), (sx - 3, sy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, col, 1, cv2.LINE_AA)
            cv2.putText(match_vis, f'QUERY @ {angle:+d}°', (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(match_vis,
                        'GRID MAP (BOX)' if used_box else 'GRID MAP',
                        (DISP + 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            ax2.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'Rotated @ {angle:+d}° ↔ Landmarks (box)\n'
                          f'{len(lm_matches)} straight connections',
                          fontsize=7, fontweight='bold', color='white')
        else:
            ax2.text(0.5, 0.5, 'No landmark\nmatches',
                     ha='center', va='center', color='gray',
                     fontsize=9, transform=ax2.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    fname = f'angle_breakdown_{q_label}.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight', facecolor='#111122')
    plt.show()
    print(f'✓ {q_label} shown + saved: {fname}')

print('\n' + '=' * 70)
print('FULL ANGLE BREAKDOWN SUMMARY')
print('=' * 70)
print(f'{"Frame":<16}{"Angle":>7}  {"Match":<12}'
      f'{"Inliers":<10}{"Score":<10}{"GPS_err":<11}{"Box?"}')
print('-' * 76)
for fr in computed:
    for ar in fr['angles']:
        best = ar['best']
        if best is None:
            print(f'{fr["label"]:<16}{ar["angle"]:>+7}°  NO MATCH')
        else:
            est_lat = float(df.iloc[best["db_idx"]][LAT_COL])
            est_lon = float(df.iloc[best["db_idx"]][LON_COL])
            gps_err = math.sqrt((fr["lat"] - est_lat) ** 2 +
                                (fr["lon"] - est_lon) ** 2) * 111000
            flag = ('✅' if gps_err < 300 else '⚠️' if gps_err < 1000 else '❌')
            box_avail = _box_image_path(best["db_idx"] + 1) is not None
            print(f'{fr["label"]:<16}{ar["angle"]:>+7}°  '
                  f'img_{best["db_idx"]+1:<8}'
                  f'{best["n_inliers"]:<10}'
                  f'{best["score"]:<10.1f}'
                  f'{gps_err:.0f}m {flag:<6}'
                  f'{"yes" if box_avail else "no"}')
    print()
