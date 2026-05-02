# ============================================================
# MODULE 6+7 — COMPLETE FINAL: PARTS A + B + C + D
# Updates:
#  A: Match lines use yellow-box sub-keypoints (green dots inside)
#  B: Test angles set to [0, 30, 45, 90, 180]
#  C: Sub-keypoint matching inside every yellow box
#  D: Live drone frame LEFT | Grid map RIGHT (100%+50%)
#     Navigate img_1→img_100 via lat/lon from 03.csv
#     Per-frame angle breakdown (incl 90°,180°) per step
#     Grayscale + Morning/Night/Midday color variants
#     All results saved to res/ folder
#
# Requires: images[], df, LAT_COL, LON_COL, KAPPA_COL,
#           valid_indices  (already loaded before this cell)
# ============================================================
import os, math
import numpy as np
import cv2
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# ── Output folders ────────────────────────────────────────────
RES_DIR       = 'res'
STEP_SAVE_DIR = os.path.join(RES_DIR, 'nav_steps')
for d in [RES_DIR, STEP_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

CELLS_DIR        = r'C:\Users\Danial Hameed\Desktop\Module\cells'
FONT             = cv2.FONT_HERSHEY_SIMPLEX
K_WORDS          = 100
TARGET_IMAGE_IDX = 75
START_IMAGE_IDX  = 0
DISPLAY_ANGLE    = 30
MIN_INLIERS_BOX  = 4
GRID_THUMB_SZ    = 128
CENTER_SZ        = 256
SHOW_EVERY_N     = 5   # Part D: plot every Nth step (1=all)

# Query rotation test angles — exactly as requested:
# 0°, 30°, 45°, 90°, 180°
TEST_ANGLES = [0, 30, 45, 90, 180]

# ============================================================
# SECTION 1 — HELPER FUNCTIONS
# ============================================================
def create_structure_mask(gray):
    blur   = cv2.GaussianBlur(gray,(5,5),0)
    edges  = cv2.Canny(blur,50,150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    return cv2.dilate(edges,kernel,iterations=2)

def extract_hybrid_orb(gray,n=2500):
    orb    = cv2.ORB_create(nfeatures=n,scaleFactor=1.2,
                             nlevels=8,edgeThreshold=10)
    mask   = create_structure_mask(gray)
    masked = cv2.bitwise_and(gray,gray,mask=mask)
    _,df_  = orb.detectAndCompute(gray,  None)
    _,dm   = orb.detectAndCompute(masked,None)
    if df_ is not None and len(df_)>1500: df_=df_[:1500]
    if dm  is not None and len(dm) >1500: dm =dm [:1500]
    if   df_ is not None and dm is not None: c=np.vstack((df_,dm))
    elif df_ is not None: c=df_
    elif dm  is not None: c=dm
    else: return np.empty((0,32),dtype=np.float32)
    if len(c)>2500: c=c[:2500]
    return c.astype(np.float32)

def desc_to_bovw(descriptors,kmeans,k):
    hist=np.zeros(k,dtype=np.float32)
    if descriptors is None or descriptors.size==0: return hist
    words=kmeans.predict(descriptors)
    hist =np.bincount(words,minlength=k).astype(np.float32)
    norm =np.linalg.norm(hist)
    if norm>0: hist/=norm
    return hist

def suppress_nearby(kps,min_dist=14,max_keep=20):
    kept=[]
    for kp in kps:
        if not any(math.sqrt((kp.pt[0]-k2.pt[0])**2+
                             (kp.pt[1]-k2.pt[1])**2)<min_dist
                   for k2 in kept):
            kept.append(kp)
        if len(kept)>=max_keep: break
    return kept

def rotate_image(img,angle):
    """
    Rotate image around its center.
    Works correctly for 0°, 30°, 45°, 90°, 180°.
    Output keeps the same (h, w) so coordinates remain comparable
    with the unrotated image — required for the H-matrix that maps
    rotated-query 800x600 -> DB 800x600.
    """
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

def color_shift(img,hue_shift=10,sat_scale=1.15,val_scale=0.85):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0]=(hsv[:,:,0]+hue_shift)%180
    hsv[:,:,1]=np.clip(hsv[:,:,1]*sat_scale,0,255)
    hsv[:,:,2]=np.clip(hsv[:,:,2]*val_scale, 0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

def transform_query(img,angle=15,hue_shift=10,
                    sat_scale=1.15,val_scale=0.85):
    return color_shift(rotate_image(img,angle),
                       hue_shift,sat_scale,val_scale)

def make_gray_bgr(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)

def make_morning(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0]=(hsv[:,:,0]+5)%180
    hsv[:,:,1]=np.clip(hsv[:,:,1]*0.8,0,255)
    hsv[:,:,2]=np.clip(hsv[:,:,2]*0.7,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

def make_night(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1]=np.clip(hsv[:,:,1]*0.3,0,255)
    hsv[:,:,2]=np.clip(hsv[:,:,2]*0.35,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

def make_midday(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1]=np.clip(hsv[:,:,1]*1.4,0,255)
    hsv[:,:,2]=np.clip(hsv[:,:,2]*1.2,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

# ============================================================
# SECTION 2 — BUILD BoVW DATABASE
# ============================================================
print('='*65)
print('STEP 1: Building ORB BoVW Database')
print('='*65)

all_desc_list=[]; per_image_desc=[]; valid_db_idx=[]
for idx in tqdm(valid_indices,desc='ORB extraction'):
    img=images[idx]
    if img is None: continue
    gray=cv2.cvtColor(cv2.resize(img,(800,600)),cv2.COLOR_BGR2GRAY)
    desc=extract_hybrid_orb(gray)
    if desc.size==0: continue
    per_image_desc.append(desc)
    all_desc_list.append(desc)
    valid_db_idx.append(idx)

all_desc_np=np.vstack(all_desc_list)
kmeans_bovw=MiniBatchKMeans(n_clusters=K_WORDS,random_state=42,
                             batch_size=2000,n_init='auto')
kmeans_bovw.fit(all_desc_np)
db_histograms=np.array(
    [desc_to_bovw(d,kmeans_bovw,K_WORDS) for d in per_image_desc],
    dtype=np.float32)
print(f'BoVW DB: {db_histograms.shape}  ✓')

# ============================================================
# SECTION 3 — GPS SPATIAL GRID
# ============================================================
GRID_SIZE=10
lats_arr=np.array([float(df.iloc[i][LAT_COL]) for i in valid_db_idx])
lons_arr=np.array([float(df.iloc[i][LON_COL]) for i in valid_db_idx])
lat_min,lat_max=lats_arr.min(),lats_arr.max()
lon_min,lon_max=lons_arr.min(),lons_arr.max()
lat_span=max(1e-9,lat_max-lat_min)
lon_span=max(1e-9,lon_max-lon_min)
spatial_cells={}
for pos,idx in enumerate(valid_db_idx):
    r=int(np.clip(((lats_arr[pos]-lat_min)/lat_span)*(GRID_SIZE-1),0,GRID_SIZE-1))
    c=int(np.clip(((lons_arr[pos]-lon_min)/lon_span)*(GRID_SIZE-1),0,GRID_SIZE-1))
    spatial_cells.setdefault(f'{r}_{c}',[]).append(pos)

def get_spatial_candidates(q_lat,q_lon,radius=2):
    r0=int(np.clip(((q_lat-lat_min)/lat_span)*(GRID_SIZE-1),0,GRID_SIZE-1))
    c0=int(np.clip(((q_lon-lon_min)/lon_span)*(GRID_SIZE-1),0,GRID_SIZE-1))
    allowed=set()
    for dr in range(-radius,radius+1):
        for dc in range(-radius,radius+1):
            for pos in spatial_cells.get(f'{r0+dr}_{c0+dc}',[]):
                allowed.add(pos)
    return list(allowed) if allowed else list(range(len(valid_db_idx)))

# ============================================================
# SECTION 4 — LANDMARK DB
# ============================================================
print('\nSTEP 2: Building Landmark DB')
orb_lm=cv2.ORB_create(nfeatures=300,scaleFactor=1.2,nlevels=6,
                        edgeThreshold=15,fastThreshold=25)
landmark_db={}
for idx in range(len(images)):
    img=images[idx]
    if img is None: continue
    img_lat=float(df.iloc[idx][LAT_COL])
    img_lon=float(df.iloc[idx][LON_COL])
    img_h,img_w=img.shape[:2]
    gray_eq=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(4,4)).apply(
        cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    kps,des=orb_lm.detectAndCompute(gray_eq,None)
    if not kps or des is None: continue
    kps_filtered=suppress_nearby(
        sorted(kps,key=lambda k:k.response,reverse=True))
    kp_all=list(kps); kept_indices=[]
    for kp_f in kps_filtered:
        for ki,kp_o in enumerate(kp_all):
            if abs(kp_o.pt[0]-kp_f.pt[0])<0.5 and \
               abs(kp_o.pt[1]-kp_f.pt[1])<0.5:
                kept_indices.append(ki); break
    lms=[]
    for lm_id,ki in enumerate(kept_indices):
        kp=kps[ki]; d=des[ki]; px,py=kp.pt
        if idx<len(df)-1:
            dlat=abs(float(df.iloc[idx+1][LAT_COL])-img_lat)
            dlon=abs(float(df.iloc[idx+1][LON_COL])-img_lon)
        else:
            dlat,dlon=0.0005,0.0005
        lms.append({'id':lm_id+1,'kp':kp,'des':d,
            'lat':img_lat+(0.5-py/img_h)*dlat*3,
            'lon':img_lon+(px/img_w-0.5)*dlon*3,
            'px':px,'py':py,'response':kp.response})
    landmark_db[idx]=lms
print(f'Landmark DB: {len(landmark_db)} images  ✓')

# ============================================================
# SECTION 5 — SHARED BOX FUNCTIONS
# UPDATED SINGLE VERSION:
#   - Yellow boxes only
#   - No green boxes
#   - ORB extracted only from content inside yellow boxes
#   - More lines for big boxes, fewer lines for small boxes
#   - Dynamic thresholds based on yellow-box size
#   - Mutual ORB matching to reduce false positives
#   - Geometry-gated using best["H"] when provided
#   - Backward compatible: old call without H will not crash
# ============================================================

YELLOW = (0, 220, 255)
GREEN  = (0, 255, 0)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

# ------------------------------------------------------------
# ORB + matcher configuration
# ------------------------------------------------------------

orb_box = cv2.ORB_create(
    nfeatures=2600,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=8,
    fastThreshold=7,
    patchSize=31
)

bf_box = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Global cap across all yellow boxes in one panel
MAX_LINES_TOTAL = 34

# Box-size-based dynamic control
AREA_RATIO_SMALL = 0.0015
AREA_RATIO_LARGE = 0.0250

# Small boxes: strict, fewer lines
MAX_LINES_SMALL = 2
MIN_LINES_SMALL = 1
ORB_DISTANCE_SMALL = 56
RATIO_TEST_SMALL = 0.72
PROJECTION_ERR_SMALL = 18.0
N_KP_SMALL = 70

# Big boxes: more lines, slightly relaxed
MAX_LINES_LARGE = 10
MIN_LINES_LARGE = 2
ORB_DISTANCE_LARGE = 84
RATIO_TEST_LARGE = 0.84
PROJECTION_ERR_LARGE = 52.0
N_KP_LARGE = 170

# Query projection margin
QUERY_POLY_MARGIN = 38

# Spatial thinning; higher = cleaner / fewer overlapping lines
DISPLAY_MIN_SPACING = 9


# ============================================================
# 5.1 — BASIC HELPERS
# ============================================================

def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _lerp(a, b, t):
    return a + (b - a) * t


def get_box_area_ratio(lm, db_img):
    """
    box area / full DB image area
    """
    db_h, db_w = db_img.shape[:2]

    box_w = max(1, int(lm["bx2"]) - int(lm["bx1"]))
    box_h = max(1, int(lm["by2"]) - int(lm["by1"]))

    box_area = float(box_w * box_h)
    full_area = float(db_w * db_h)

    return box_area / max(1.0, full_area)


def get_dynamic_box_params(lm, db_img):
    """
    Larger yellow box:
      - more ORB keypoints
      - more lines
      - looser distance / ratio / projection thresholds

    Smaller yellow box:
      - fewer lines
      - stricter thresholds
      - fewer false matches
    """
    area_ratio = get_box_area_ratio(lm, db_img)

    t = (area_ratio - AREA_RATIO_SMALL) / max(1e-9, AREA_RATIO_LARGE - AREA_RATIO_SMALL)
    t = _clip01(t)

    return {
        "area_ratio": area_ratio,
        "max_lines": int(round(_lerp(MAX_LINES_SMALL, MAX_LINES_LARGE, t))),
        "min_lines": int(round(_lerp(MIN_LINES_SMALL, MIN_LINES_LARGE, t))),
        "orb_distance_limit": float(_lerp(ORB_DISTANCE_SMALL, ORB_DISTANCE_LARGE, t)),
        "ratio_test": float(_lerp(RATIO_TEST_SMALL, RATIO_TEST_LARGE, t)),
        "proj_err_max": float(_lerp(PROJECTION_ERR_SMALL, PROJECTION_ERR_LARGE, t)),
        "n_kp": int(round(_lerp(N_KP_SMALL, N_KP_LARGE, t))),
    }


# ============================================================
# 5.2 — LOAD YELLOW BOX IMAGE
# ============================================================

def load_cell_box_image(img_idx_1based):
    """
    Load yellow-box image from CELLS_DIR.

    Expected names:
      cell_001_boxes.jpg / jpeg / png
      ...
      cell_100_boxes.jpg / jpeg / png
    """
    candidates = [
        os.path.join(CELLS_DIR, f"cell_{img_idx_1based:03d}_boxes.jpg"),
        os.path.join(CELLS_DIR, f"cell_{img_idx_1based:03d}_boxes.jpeg"),
        os.path.join(CELLS_DIR, f"cell_{img_idx_1based:03d}_boxes.png"),
    ]

    for p in candidates:
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None and img.size > 0:
                return img

    return None


# ============================================================
# 5.3 — DETECT YELLOW BOXES FROM IMAGE
# ============================================================

def detect_yellow_boxes_from_box_image(box_img):
    """
    Fallback if cell_XXX_landmarks.csv is missing.
    Detect yellow boxes directly from cell_XXX_boxes image.
    """
    if box_img is None or box_img.size == 0:
        return []

    hsv = cv2.cvtColor(box_img, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 60, 60], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = box_img.shape[:2]
    boxes = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw * bh < 700:
            continue

        if bw < 18 or bh < 18:
            continue

        if bw > 0.98 * w or bh > 0.98 * h:
            continue

        boxes.append((x, y, x + bw, y + bh))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # Remove duplicate / overlapping boxes
    filtered = []

    for b in boxes:
        x1, y1, x2, y2 = b
        keep = True

        for fb in filtered:
            fx1, fy1, fx2, fy2 = fb

            ix1 = max(x1, fx1)
            iy1 = max(y1, fy1)
            ix2 = min(x2, fx2)
            iy2 = min(y2, fy2)

            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (x2 - x1) * (y2 - y1)
            a2 = (fx2 - fx1) * (fy2 - fy1)

            iou = inter / max(1, a1 + a2 - inter)

            if iou > 0.45:
                keep = False
                break

        if keep:
            filtered.append(b)

    return filtered


# ============================================================
# 5.4 — LOAD LANDMARK CSV OR FALLBACK TO DETECTED BOXES
# ============================================================

def load_cell_landmarks(img_idx_1based):
    """
    Preferred:
      cell_XXX_landmarks.csv

    Fallback:
      detect yellow boxes from cell_XXX_boxes image.

    Returns landmark dicts with:
      lm_id, bx1, by1, bx2, by2, cx, cy, box_w, box_h
    """
    p = os.path.join(CELLS_DIR, f"cell_{img_idx_1based:03d}_landmarks.csv")

    if os.path.exists(p):
        lm_df = pd.read_csv(p)
        out = []

        for k, row in lm_df.iterrows():
            bx1 = int(row["bx1"])
            by1 = int(row["by1"])
            bx2 = int(row["bx2"])
            by2 = int(row["by2"])

            out.append({
                "lm_id": str(row.get("landmark_id", row.get("lm_id", row.get("num", k + 1)))),
                "num": int(row.get("num", k + 1)),
                "lat": float(row.get("lat", float(df.iloc[img_idx_1based - 1][LAT_COL]))),
                "lon": float(row.get("lon", float(df.iloc[img_idx_1based - 1][LON_COL]))),
                "bx1": bx1,
                "by1": by1,
                "bx2": bx2,
                "by2": by2,
                "cx": int(row.get("cx", (bx1 + bx2) // 2)),
                "cy": int(row.get("cy", (by1 + by2) // 2)),
                "box_w": int(row.get("box_w", bx2 - bx1)),
                "box_h": int(row.get("box_h", by2 - by1)),
            })

        return out

    # Fallback from cell_XXX_boxes image
    box_img = load_cell_box_image(img_idx_1based)

    if box_img is None:
        return []

    db_img = images[img_idx_1based - 1]

    if db_img is None:
        return []

    raw_boxes = detect_yellow_boxes_from_box_image(box_img)

    bh, bw = box_img.shape[:2]
    dh, dw = db_img.shape[:2]

    sx = dw / bw
    sy = dh / bh

    out = []

    for k, (x1, y1, x2, y2) in enumerate(raw_boxes):
        bx1 = int(x1 * sx)
        by1 = int(y1 * sy)
        bx2 = int(x2 * sx)
        by2 = int(y2 * sy)

        out.append({
            "lm_id": f"{k + 1}a",
            "num": k + 1,
            "lat": float(df.iloc[img_idx_1based - 1][LAT_COL]),
            "lon": float(df.iloc[img_idx_1based - 1][LON_COL]),
            "bx1": bx1,
            "by1": by1,
            "bx2": bx2,
            "by2": by2,
            "cx": (bx1 + bx2) // 2,
            "cy": (by1 + by2) // 2,
            "box_w": bx2 - bx1,
            "box_h": by2 - by1,
        })

    return out


# ============================================================
# 5.5 — IMAGE PREPROCESSING FOR ORB
# ============================================================

def clahe_gray_box(img_bgr):
    """
    Convert BGR image to contrast-enhanced grayscale.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(4, 4)
    ).apply(gray)


def structure_mask_box(gray):
    """
    Focus ORB on roofs, paths, roads, edges and building shapes.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 155)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(edges, kernel, iterations=1)

    return mask


# ============================================================
# 5.6 — EXTRACT ORB ONLY INSIDE ONE YELLOW BOX
# ============================================================

def extract_sub_keypoints_in_box(img_bgr, box, n_kp=150):
    """
    Extract ORB keypoints only inside the yellow box.
    Returned keypoints are mapped back to original DB-image coordinates.

    No green boxes.
    No full-image DB ORB.
    """
    bx1, by1, bx2, by2 = map(int, box)

    crop = img_bgr[by1:by2, bx1:bx2]

    if crop is None or crop.size == 0:
        return None, None

    ch, cw = crop.shape[:2]

    # Upscale small boxes so ORB has enough structure
    scale = max(1.0, 170.0 / max(1, min(ch, cw)))

    if scale > 1:
        crop_used = cv2.resize(
            crop,
            (int(cw * scale), int(ch * scale)),
            interpolation=cv2.INTER_CUBIC
        )
    else:
        crop_used = crop.copy()

    gray = clahe_gray_box(crop_used)
    mask = structure_mask_box(gray)

    kps, des = orb_box.detectAndCompute(gray, mask)

    if not kps or des is None:
        kps, des = orb_box.detectAndCompute(gray, None)

    if not kps or des is None:
        return None, None

    order = sorted(range(len(kps)), key=lambda i: kps[i].response, reverse=True)
    order = order[:n_kp]

    kps = [kps[i] for i in order]
    des = des[order]

    # Convert crop coordinates back to original DB-image coordinates
    kps_full = [
        cv2.KeyPoint(
            kp.pt[0] / scale + bx1,
            kp.pt[1] / scale + by1,
            kp.size,
            kp.angle,
            kp.response,
            kp.octave
        )
        for kp in kps
    ]

    return kps_full, des


def extract_query_orb(q_img):
    """
    Extract ORB from query image.
    """
    gray = clahe_gray_box(q_img)
    kp_q, des_q = orb_box.detectAndCompute(gray, None)
    return kp_q, des_q


# ============================================================
# 5.7 — GEOMETRY HELPERS
# ============================================================

def db_point_to_800(pt_db, db_img):
    """
    Original DB-image point -> 800x600 DB coordinate system.
    Needed because best['H'] maps query 800x600 -> DB 800x600.
    """
    db_h, db_w = db_img.shape[:2]

    return np.array([
        pt_db[0] * 800.0 / db_w,
        pt_db[1] * 600.0 / db_h
    ], dtype=np.float32)


def yellow_box_to_db800_poly(lm, db_img):
    """
    Convert one yellow box from original DB-image coords to 800x600 DB coords.
    """
    db_h, db_w = db_img.shape[:2]

    sx = 800.0 / db_w
    sy = 600.0 / db_h

    bx1 = lm["bx1"] * sx
    by1 = lm["by1"] * sy
    bx2 = lm["bx2"] * sx
    by2 = lm["by2"] * sy

    return np.float32([
        [bx1, by1],
        [bx2, by1],
        [bx2, by2],
        [bx1, by2]
    ])


def expand_poly(poly, margin=QUERY_POLY_MARGIN):
    """
    Expand polygon outward around centroid.
    """
    c = np.mean(poly, axis=0)
    out = []

    for p in poly:
        v = p - c
        norm = np.linalg.norm(v)

        if norm < 1e-6:
            out.append(p)
        else:
            out.append(p + margin * v / norm)

    return np.float32(out)


def query_poly_for_yellow_box(lm, db_img, H_q_to_db):
    """
    Inverse-project DB yellow box into query coordinates.
    H_q_to_db maps query 800x600 -> DB 800x600.
    """
    if H_q_to_db is None:
        return None

    try:
        H_inv = np.linalg.inv(H_q_to_db)
    except Exception:
        return None

    db_poly = yellow_box_to_db800_poly(lm, db_img).reshape(-1, 1, 2)

    try:
        q_poly = cv2.perspectiveTransform(db_poly, H_inv)
    except Exception:
        return None

    q_poly = q_poly.reshape(-1, 2)
    q_poly = expand_poly(q_poly, margin=QUERY_POLY_MARGIN)

    return q_poly


def point_inside_poly(pt, poly):
    """
    Test if point lies inside projected yellow-box area.
    """
    if poly is None:
        return True

    return cv2.pointPolygonTest(
        poly.astype(np.float32),
        (float(pt[0]), float(pt[1])),
        False
    ) >= 0


def project_query_point_to_db800(pt_q, H_q_to_db):
    """
    Project query point to DB 800x600 coordinate system.
    """
    p = np.float32([[pt_q]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(p, H_q_to_db)

    return proj.reshape(-1, 2)[0]


# ============================================================
# 5.8 — MUTUAL ORB MATCHING + DISPLAY FILTER
# ============================================================

def mutual_ratio_matches(des_db, des_q_sub, ratio_test=0.78, dist_limit=80):
    """
    Mutual KNN ratio-test matching.

    This removes many false matches compared with one-way matching.
    """
    if des_db is None or des_q_sub is None:
        return []

    if len(des_db) < 2 or len(des_q_sub) < 2:
        return []

    try:
        fwd = bf_box.knnMatch(des_db, des_q_sub, k=2)
        rev = bf_box.knnMatch(des_q_sub, des_db, k=2)
    except Exception:
        return []

    rev_best = {}

    for pair in rev:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance > dist_limit:
            continue

        if m.distance >= ratio_test * n.distance:
            continue

        # key: query-desc index in des_q_sub, train-desc index in des_db
        rev_best[(m.queryIdx, m.trainIdx)] = m

    good = []

    for pair in fwd:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance > dist_limit:
            continue

        if m.distance >= ratio_test * n.distance:
            continue

        # fwd m: queryIdx = db desc, trainIdx = q_sub desc
        # mutual means reverse has q_sub -> db
        if (m.trainIdx, m.queryIdx) in rev_best:
            good.append(m)

    return good


def filter_matches_by_box_size(kp_db, kp_q, matches, max_keep, min_spacing=DISPLAY_MIN_SPACING):
    """
    Keep strongest spatially separated matches only.
    Prevents dense clutter.
    """
    if not matches:
        return []

    matches = sorted(matches, key=lambda m: m.distance)

    kept = []
    used = []

    for m in matches:
        dbx, dby = kp_db[m.queryIdx].pt
        qx, qy = kp_q[m.trainIdx].pt

        ok = True

        for udbx, udby, uqx, uqy in used:
            if math.hypot(dbx - udbx, dby - udby) < min_spacing:
                ok = False
                break

            if math.hypot(qx - uqx, qy - uqy) < min_spacing:
                ok = False
                break

        if ok:
            kept.append(m)
            used.append((dbx, dby, qx, qy))

        if len(kept) >= max_keep:
            break

    return kept


# ============================================================
# 5.9 — MATCH ONE YELLOW BOX TO QUERY
# ============================================================

def match_one_yellow_box_geogated(db_img, q_img, lm, H_q_to_db, kp_q, des_q):
    """
    Match one yellow box only.

    Big box:
      - more keypoints
      - more allowed lines
      - slightly relaxed thresholds

    Small box:
      - fewer lines
      - stricter thresholds
      - fewer false matches
    """
    box = (lm["bx1"], lm["by1"], lm["bx2"], lm["by2"])
    dyn = get_dynamic_box_params(lm, db_img)

    kp_db, des_db = extract_sub_keypoints_in_box(
        db_img,
        box,
        n_kp=dyn["n_kp"]
    )

    if kp_db is None or des_db is None or des_q is None or not kp_q:
        return None

    # Default: all query keypoints
    q_indices = list(range(len(kp_q)))

    # If H is available, restrict query keypoints to projected yellow-box area
    if H_q_to_db is not None:
        q_poly = query_poly_for_yellow_box(lm, db_img, H_q_to_db)

        q_indices_poly = []

        for i, kp in enumerate(kp_q):
            if point_inside_poly(kp.pt, q_poly):
                q_indices_poly.append(i)

        if len(q_indices_poly) >= 8:
            q_indices = q_indices_poly

    des_q_sub = des_q[q_indices]

    if des_q_sub is None or len(des_q_sub) < 4:
        return None

    # Mutual ORB matching
    raw_good = mutual_ratio_matches(
        des_db,
        des_q_sub,
        ratio_test=dyn["ratio_test"],
        dist_limit=dyn["orb_distance_limit"]
    )

    if len(raw_good) < dyn["min_lines"]:
        return None

    candidates = []

    for m in raw_good:
        original_q_idx = q_indices[m.trainIdx]

        fixed_m = cv2.DMatch(
            _queryIdx=m.queryIdx,
            _trainIdx=original_q_idx,
            _imgIdx=0,
            _distance=m.distance
        )

        # Geometry check only if H is available
        if H_q_to_db is not None:
            p_q = kp_q[original_q_idx].pt
            p_db = kp_db[m.queryIdx].pt

            try:
                p_q_db800 = project_query_point_to_db800(p_q, H_q_to_db)
                p_db800 = db_point_to_800(p_db, db_img)
                err = float(np.linalg.norm(p_q_db800 - p_db800))
            except Exception:
                err = 9999.0

            if err > dyn["proj_err_max"]:
                continue

        candidates.append(fixed_m)

    if len(candidates) < dyn["min_lines"]:
        return None

    # Final line selection according to box size
    candidates = filter_matches_by_box_size(
        kp_db,
        kp_q,
        candidates,
        max_keep=dyn["max_lines"],
        min_spacing=DISPLAY_MIN_SPACING
    )

    if len(candidates) < dyn["min_lines"]:
        return None

    return {
        "lm_id": lm["lm_id"],
        "kp_db": kp_db,
        "kp_q": kp_q,
        "matches": candidates,
        "n_inliers": len(candidates),
        "n_good": len(raw_good),
        "score": float(sum(100 - m.distance for m in candidates)),
        "area_ratio": dyn["area_ratio"],
        "max_lines_box": dyn["max_lines"],
        "min_lines_box": dyn["min_lines"],
        "orb_distance_limit": dyn["orb_distance_limit"],
        "ratio_test": dyn["ratio_test"],
        "proj_err_max": dyn["proj_err_max"],
    }


# ============================================================
# 5.10 — MATCH ALL YELLOW BOXES TO QUERY
# ============================================================

def match_yellow_box_content_to_query(db_img, q_img, landmarks, H_q_to_db=None):
    """
    Match only content inside yellow boxes to query.

    Big boxes are processed first because they support more reliable matching.
    """
    if db_img is None or q_img is None or not landmarks:
        return {}, [], landmarks, {}

    kp_q, des_q = extract_query_orb(q_img)

    if des_q is None or not kp_q:
        return {}, [], landmarks, {}

    # Process larger boxes first
    landmarks_sorted = sorted(
        landmarks,
        key=lambda lm: (int(lm["bx2"]) - int(lm["bx1"])) * (int(lm["by2"]) - int(lm["by1"])),
        reverse=True
    )

    match_results = {}
    matched = []
    unmatched = []

    all_kp_db = []
    all_matches_global = []

    total_lines = 0

    for lm in landmarks_sorted:
        result = match_one_yellow_box_geogated(
            db_img=db_img,
            q_img=q_img,
            lm=lm,
            H_q_to_db=H_q_to_db,
            kp_q=kp_q,
            des_q=des_q
        )

        if result is None:
            match_results[lm["lm_id"]] = None
            unmatched.append(lm)
            continue

        offset = len(all_kp_db)
        all_kp_db.extend(result["kp_db"])

        remapped = []

        for m in result["matches"]:
            remapped.append(
                cv2.DMatch(
                    _queryIdx=m.queryIdx + offset,
                    _trainIdx=m.trainIdx,
                    _imgIdx=0,
                    _distance=m.distance
                )
            )

        result["matches_global"] = remapped

        match_results[lm["lm_id"]] = result
        matched.append(lm)

        all_matches_global.extend(remapped)
        total_lines += len(remapped)

        if total_lines >= MAX_LINES_TOTAL:
            break

    all_matches_global = all_matches_global[:MAX_LINES_TOTAL]

    pack = {
        "kp_db": all_kp_db,
        "kp_q": kp_q,
        "matches": all_matches_global
    }

    return match_results, matched, unmatched, pack


# ============================================================
# 5.11 — ANNOTATE DB IMAGE WITH YELLOW BOXES
# ============================================================

def annotate_db_boxes(db_img, landmarks, match_results=None):
    """
    Draw yellow boxes only.
    No green boxes and no green dots.

    If a box is matched, show n/max_lines in yellow text.
    """
    vis = db_img.copy()
    h, w = vis.shape[:2]

    fs = max(0.45, min(w, h) / 1400)
    thick = max(2, min(w, h) // 300)

    for lm in landmarks:
        bx1, by1, bx2, by2 = int(lm["bx1"]), int(lm["by1"]), int(lm["bx2"]), int(lm["by2"])
        lm_id = lm["lm_id"]

        cv2.rectangle(vis, (bx1, by1), (bx2, by2), YELLOW, thick)

        (tw, th), _ = cv2.getTextSize(str(lm_id), FONT, fs, 2)

        tx1 = max(0, bx2 - tw - 6)
        ty2 = min(h - 1, by1 + th + 8)

        cv2.rectangle(vis, (tx1, by1), (bx2, ty2), BLACK, -1)

        cv2.putText(
            vis,
            str(lm_id),
            (tx1 + 2, ty2 - 4),
            FONT,
            fs,
            YELLOW,
            2,
            cv2.LINE_AA
        )

        if match_results is not None and match_results.get(lm_id) is not None:
            r = match_results[lm_id]
            n = r["n_inliers"]
            max_box_lines = r.get("max_lines_box", n)

            cv2.putText(
                vis,
                f"{n}/{max_box_lines}",
                (bx1 + 4, min(h - 8, by2 + 18)),
                FONT,
                fs * 0.8,
                YELLOW,
                1,
                cv2.LINE_AA
            )

    return vis


# ============================================================
# 5.12 — DRAW MATCHES IN COLUMN 3
# ============================================================

def draw_landmark_matches_col_boxes(ax2, q_img, match_img,
                                    best_idx, lms_csv, H_q_to_db=None,
                                    DISP=300):
    """
    Part A col-3:
      left  = best DB image
      right = query / winning angled query
      lines = ORB matches from yellow-box content only

    For matching to be perfect across rotated queries (0°, 30°,
    45°, 90°, 180°), the caller MUST pass:
      - q_img         = the rotated query image at the WINNING angle
                        (i.e. best['rotated_query_bgr'])
      - H_q_to_db     = best['H'] (which was estimated against that
                        rotated query). Otherwise yellow-box ORB
                        matching against a non-rotated query will
                        not align with the H matrix.
    """
    if match_img is None or not lms_csv:
        ax2.axis("off")
        return {}, [], []

    match_results, matched, unmatched, pack = match_yellow_box_content_to_query(
        db_img=match_img,
        q_img=q_img,
        landmarks=lms_csv,
        H_q_to_db=H_q_to_db
    )

    if not pack or "matches" not in pack or len(pack["matches"]) == 0:
        msg = (
            "NO GEOMETRY-GATED\nYELLOW-BOX ORB MATCH"
            if H_q_to_db is not None
            else "NO YELLOW-BOX\nORB MATCH"
        )

        ax2.text(
            0.5, 0.5,
            msg,
            ha="center",
            va="center",
            color="red",
            fontsize=10,
            fontweight="bold",
            transform=ax2.transAxes
        )
        ax2.axis("off")
        return match_results, matched, unmatched

    kp_db = pack["kp_db"]
    kp_q = pack["kp_q"]
    matches = pack["matches"]

    db_h, db_w = match_img.shape[:2]
    q_h, q_w = q_img.shape[:2]

    db_d = cv2.resize(match_img, (DISP, DISP))
    q_d = cv2.resize(q_img, (DISP, DISP))

    sx_d = DISP / db_w
    sy_d = DISP / db_h
    sx_q = DISP / q_w
    sy_q = DISP / q_h

    kp_db_sc = [
        cv2.KeyPoint(
            kp.pt[0] * sx_d,
            kp.pt[1] * sy_d,
            kp.size,
            kp.angle,
            kp.response,
            kp.octave
        )
        for kp in kp_db
    ]

    kp_q_sc = [
        cv2.KeyPoint(
            kp.pt[0] * sx_q,
            kp.pt[1] * sy_q,
            kp.size,
            kp.angle,
            kp.response,
            kp.octave
        )
        for kp in kp_q
    ]

    mv = cv2.drawMatches(
        db_d,
        kp_db_sc,
        q_d,
        kp_q_sc,
        matches,
        None,
        matchColor=GREEN,
        singlePointColor=(90, 255, 90),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Draw yellow boxes on DB side only
    for lm in lms_csv:
        cv2.rectangle(
            mv,
            (int(lm["bx1"] * sx_d), int(lm["by1"] * sy_d)),
            (int(lm["bx2"] * sx_d), int(lm["by2"] * sy_d)),
            YELLOW,
            1
        )

    label = (
        "DB YELLOW-BOX CONTENT | H-GATED"
        if H_q_to_db is not None
        else "DB YELLOW-BOX CONTENT | ORB ONLY"
    )

    cv2.putText(
        mv,
        label,
        (5, 15),
        FONT,
        0.34,
        YELLOW,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        mv,
        "QUERY",
        (DISP + 5, 15),
        FONT,
        0.36,
        WHITE,
        1,
        cv2.LINE_AA
    )

    ax2.imshow(cv2.cvtColor(mv, cv2.COLOR_BGR2RGB))
    ax2.set_title(
        f"Yellow-box content ↔ query\n{len(matches)} ORB lines",
        fontsize=7,
        fontweight="bold"
    )
    ax2.axis("off")

    return match_results, matched, unmatched

# ============================================================
# SECTION 6 — RANSAC + MULTI-ANGLE
# ============================================================
def _ransac_match_single(gray_q,filtered_pos,scores_cos):
    orb_v =cv2.ORB_create(nfeatures=2500,scaleFactor=1.2,
                           nlevels=8,edgeThreshold=10)
    bf_ham=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
    kp_q,des_q=orb_v.detectAndCompute(gray_q,None)
    if des_q is None or len(kp_q)<4: return None
    best=None; best_score=-1
    for pos in filtered_pos:
        db_idx=valid_db_idx[pos]; db_img=images[db_idx]
        if db_img is None: continue
        gray_db=cv2.cvtColor(cv2.resize(db_img,(800,600)),cv2.COLOR_BGR2GRAY)
        kp_db,des_db=orb_v.detectAndCompute(gray_db,None)
        if des_db is None or len(kp_db)<4: continue
        raw=bf_ham.knnMatch(des_q,des_db,k=2)
        good=[m for pair in raw if len(pair)==2
              for m,n in [pair] if m.distance<0.75*n.distance]
        if len(good)<4: continue
        src=np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([kp_db[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        if H is None or mask is None: continue
        mask_b=mask.ravel().astype(bool); n_inliers=int(mask_b.sum())
        if n_inliers<4: continue
        score=0.7*n_inliers+0.3*len(good)
        if score>best_score:
            best_score=score
            best={'db_idx':db_idx,'bovw_score':float(scores_cos[pos]),
                  'n_inliers':n_inliers,'n_good':len(good),'score':score,
                  'inlier_ratio':n_inliers/(len(good)+1e-6),
                  'inlier_matches':[m for m,k in zip(good,mask_b) if k],
                  'kp_q':kp_q,'kp_db':kp_db,
                  'gray_q':gray_q,'gray_db':gray_db,'H':H}
    return best

def find_best_match_multiangle(query_img_bgr,q_lat,q_lon,
                                top_k_bovw=10,ransac_radius=2):
    """
    Try every angle in TEST_ANGLES = [0, 30, 45, 90, 180] on the
    query image, then RANSAC-match each rotated query against the
    BoVW-shortlisted DB images, and keep the absolute best result.

    The winning rotated query is stored in
        global_best['rotated_query_bgr']
    so that the yellow-box ORB matching in Part A can match the
    same coordinate frame as best['H'] (which is estimated against
    the rotated query, not the original query). This is what makes
    matching against the box images "perfect" across all five
    angles.
    """
    global_best=None; global_best_score=-1; global_best_angle=0
    per_angle_results={}
    for angle in TEST_ANGLES:
        rotated=rotate_image(cv2.resize(query_img_bgr,(800,600)),angle)
        gray_q =cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
        desc_q =extract_hybrid_orb(gray_q)
        hist_q =desc_to_bovw(desc_q,kmeans_bovw,K_WORDS).reshape(1,-1)
        scores_cos=cos_sim(hist_q,db_histograms)[0]
        ranked_pos=np.argsort(scores_cos)[::-1][:top_k_bovw]
        spatial_pos=set(get_spatial_candidates(q_lat,q_lon,radius=ransac_radius))
        filtered_pos=([p for p in ranked_pos if p in spatial_pos] or list(ranked_pos))
        result=_ransac_match_single(gray_q,filtered_pos,scores_cos)
        if result is not None:
            result['winning_angle']=angle
            result['rotated_query_bgr']=rotated
            per_angle_results[angle]={
                'db_idx':result['db_idx'],
                'n_inliers':result['n_inliers'],
                'score':result['score']
            }
            if result['score']>global_best_score:
                global_best_score=result['score']; global_best=result
                global_best_angle=angle
        else:
            per_angle_results[angle]=None
    if global_best is not None:
        global_best['per_angle']=per_angle_results
        print(f'  Best angle:{global_best_angle}°  '
              f'img_{global_best["db_idx"]+1}  '
              f'inliers={global_best["n_inliers"]}  '
              f'score={global_best["score"]:.1f}')
        # Per-angle breakdown
        for a in TEST_ANGLES:
            r=per_angle_results.get(a)
            if r is None:
                print(f'    angle {a:>4}°: no match')
            else:
                print(f'    angle {a:>4}°: img_{r["db_idx"]+1}  '
                      f'in={r["n_inliers"]}  sc={r["score"]:.1f}')
    return global_best

# ============================================================
# SECTION 7 — SIMULATE UAV FRAMES
# ============================================================
nav_path=list(range(START_IMAGE_IDX,TARGET_IMAGE_IDX+1))
sim_frames=[]
for k in range(len(nav_path)-1):
    i,j=nav_path[k],nav_path[k+1]
    img_a,img_b=images[i],images[j]
    if img_a is None or img_b is None: continue
    h=min(img_a.shape[0],img_b.shape[0]); w=min(img_a.shape[1],img_b.shape[1])
    sim_frames.append({'type':'real','idx':i,'image':images[i],
        'lat':float(df.iloc[i][LAT_COL]),'lon':float(df.iloc[i][LON_COL]),
        'kappa':float(df.iloc[i][KAPPA_COL]) if KAPPA_COL else 0.0,
        'label':f'img_{i+1}'})
    blended=cv2.addWeighted(cv2.resize(img_a,(w,h)),0.5,
                             cv2.resize(img_b,(w,h)),0.5,0)
    sim_frames.append({'type':'synthetic','idx':None,'image':blended,
        'lat':(float(df.iloc[i][LAT_COL])+float(df.iloc[j][LAT_COL]))/2,
        'lon':(float(df.iloc[i][LON_COL])+float(df.iloc[j][LON_COL]))/2,
        'kappa':float(df.iloc[i][KAPPA_COL]) if KAPPA_COL else 0.0,
        'label':f'syn_{i+1}_{j+1}'})
sim_frames.append({'type':'real','idx':TARGET_IMAGE_IDX,
    'image':images[TARGET_IMAGE_IDX],
    'lat':float(df.iloc[TARGET_IMAGE_IDX][LAT_COL]),
    'lon':float(df.iloc[TARGET_IMAGE_IDX][LON_COL]),
    'kappa':float(df.iloc[TARGET_IMAGE_IDX][KAPPA_COL]) if KAPPA_COL else 0.0,
    'label':f'img_{TARGET_IMAGE_IDX+1}'})
print(f'Simulated frames: {len(sim_frames)}  ✓')

# ============================================================
# SECTION 8 — PART A: BEST MATCH SUMMARY
# Test angles applied to the QUERY image: 0°, 30°, 45°, 90°, 180°
# Yellow-box matching on col-3 uses the WINNING-angle rotated
# query and the H matrix that was estimated against it, so the
# match lines align perfectly with the box images.
# ============================================================
test_frame_indices=[0,10,20,30,40,50,60,70,-1]
fig_rows=len(test_frame_indices)
fig,axes=plt.subplots(fig_rows,3,figsize=(22,fig_rows*4))
localization_results=[]

print('\n'+'='*70)
print(f'PART A — BEST MATCH  Angles:{TEST_ANGLES}')
print('Col 3: yellow box sub-keypoint match lines')
print('='*70)

for row_i,fi in enumerate(test_frame_indices):
    frame=sim_frames[fi]; q_img=frame['image']
    q_lat,q_lon=frame['lat'],frame['lon']
    q_label,q_type=frame['label'],frame['type']
    if q_img is None: continue
    q_display=transform_query(cv2.resize(q_img,(400,400)),angle=DISPLAY_ANGLE)
    print(f'\n[{q_label}]')
    best=find_best_match_multiangle(q_img,q_lat,q_lon,top_k_bovw=10,ransac_radius=2)
    if best is None:
        for ax in axes[row_i]: ax.axis('off'); continue
    best_idx=best['db_idx']; n_inliers=best['n_inliers']
    ratio=best['inlier_ratio']; score=best['score']; win_angle=best['winning_angle']
    est_lat=float(df.iloc[best_idx][LAT_COL]); est_lon=float(df.iloc[best_idx][LON_COL])
    gps_err=math.sqrt((q_lat-est_lat)**2+(q_lon-est_lon)**2)*111000
    match_img=images[best_idx]
    localization_results.append({'frame_label':q_label,'frame_type':q_type,
        'true_lat':q_lat,'true_lon':q_lon,'est_lat':est_lat,'est_lon':est_lon,
        'best_match':best_idx,'n_inliers':n_inliers,'inlier_ratio':ratio,
        'score':score,'winning_angle':win_angle,'gps_err_m':gps_err})

    # ── Col 1: query (display-rotated for visualization only)
    axes[row_i,0].imshow(cv2.cvtColor(q_display,cv2.COLOR_BGR2RGB))
    axes[row_i,0].set_title(f'Query:{q_label} [{q_type}]\n'
        f'Lat={q_lat:.5f} Lon={q_lon:.5f}\n[won {win_angle}°]',
        fontsize=7,fontweight='bold'); axes[row_i,0].axis('off')

    # ── Col 2: matched DB image with yellow boxes + H projection
    if match_img is not None:
        lms_csv=load_cell_landmarks(best_idx+1)
        m_vis=annotate_db_boxes(match_img,lms_csv,{})
        m_vis=cv2.resize(m_vis,(400,400))
        h_q,w_q=best['gray_q'].shape
        corners=np.float32([[0,0],[0,h_q-1],[w_q-1,h_q-1],[w_q-1,0]]).reshape(-1,1,2)
        proj=cv2.perspectiveTransform(corners,best['H'])
        proj[:,0,0]*=400/800; proj[:,0,1]*=400/600
        cv2.polylines(m_vis,[np.int32(proj)],True,(255,255,0),2,cv2.LINE_AA)
        axes[row_i,1].imshow(cv2.cvtColor(m_vis,cv2.COLOR_BGR2RGB))
    tc=('darkgreen' if gps_err<300 else 'darkorange' if gps_err<1000 else 'darkred')
    axes[row_i,1].set_title(f'Best:img_{best_idx+1} [{win_angle}°]\n'
        f'In={n_inliers} Sc={score:.1f} GPS≈{gps_err:.0f}m',
        fontsize=7,fontweight='bold',color=tc); axes[row_i,1].axis('off')

    # ── Col 3: yellow-box ORB matching against the WINNING rotated query
    # Critical fix for "match perfectly with the box images":
    # use the rotated 800x600 query and best['H'] together so the
    # H-gated yellow-box ORB matching works for every angle in
    # TEST_ANGLES = [0, 30, 45, 90, 180].
    lms_csv=load_cell_landmarks(best_idx+1)
    rotated_query_bgr=best.get('rotated_query_bgr')
    if rotated_query_bgr is None:
        rotated_query_bgr=rotate_image(cv2.resize(q_img,(800,600)),win_angle)
    draw_landmark_matches_col_boxes(
        axes[row_i,2],
        rotated_query_bgr,
        match_img,
        best_idx,
        lms_csv,
        H_q_to_db=best['H']
    )

plt.suptitle('PART A — ORB Multi-Angle Best Match  '
             f'(angles tested: {TEST_ANGLES})\n'
             'Col 3: yellow box sub-keypoints → straight match lines '
             '(matched on the winning-angle rotated query)',
             fontsize=11,fontweight='bold')
plt.tight_layout()
out_a=os.path.join(RES_DIR,'localization_multiangle.png')
plt.savefig(out_a,dpi=120,bbox_inches='tight'); plt.show()
print(f'PART A → {out_a}')

print(f'\n{"Frame":<18}{"Match":<10}{"Angle":<8}{"Inliers":<10}'
      f'{"Ratio":<8}{"Score":<10}{"GPS_err"}')
print('-'*70)
for r in localization_results:
    flag='✅' if r['gps_err_m']<300 else '⚠️' if r['gps_err_m']<1000 else '❌'
    print(f'{r["frame_label"]:<18}img_{r["best_match"]+1:<6}'
          f'{r["winning_angle"]:>+4}°   {r["n_inliers"]:<10}'
          f'{r["inlier_ratio"]:<8.2f}{r["score"]:<10.1f}'
          f'{r["gps_err_m"]:.0f}m {flag}')
