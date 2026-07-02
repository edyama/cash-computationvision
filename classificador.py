"""
classificador.py – Autenticidade de cédulas BRL.

Técnicas: GRAY · SOBEL · LAPLACE · FFT · LBP · HIST_COR · ORB · HOG · FACE · COMBINADO · SVM
Treinamento: python classificador.py --treinar --verdadeiras <pasta> --falsas <pasta>

Correção: vetor de features tem tamanho FIXO independente das libs instaladas,
evitando ValueError de incompatibilidade com modelos salvos anteriormente.
"""

import os
import cv2 as cv
import pickle
import logging
import numpy as np
from typing import List

log = logging.getLogger(__name__)

# ── Importações opcionais ──────────────────────────────────────────
try:
    from skimage.feature import local_binary_pattern, hog as sk_hog
    from skimage.measure import shannon_entropy
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

try:
    import face_recognition; FACE_OK = True
except ImportError:
    FACE_OK = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import joblib; SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Caminhos ──────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
PATH_SVM = os.path.join(_DIR, "modelos", "svm_notas.pkl")
PATH_TPL = os.path.join(_DIR, "modelos", "template_orb.pkl")
os.makedirs(os.path.join(_DIR, "modelos"), exist_ok=True)

MODOS = ["COMBINADO","GRAY","SOBEL","LAPLACE","FFT","LBP","HIST_COR","ORB","HOG","FACE","SVM"]
_PESOS = {"GRAY":.10,"SOBEL":.15,"LAPLACE":.15,"FFT":.15,"LBP":.10,
          "HIST_COR":.10,"ORB":.10,"HOG":.10,"FACE":.05}

# ── Tamanhos FIXOS de cada bloco do vetor de features ─────────────
# Estes valores nunca mudam, independente de libs instaladas.
# HOG: imagem 128×256, 9 orientações, célula 16×16, bloco 2×2 → 3780
# LBP: 18 bins (uniform P=16)
# HSV: 3 canais × 32 bins = 96
_HOG_SIZE  = 3780
_LBP_SIZE  = 18
_HSV_SIZE  = 96          # 3 × 32
_FEAT_SIZE = 3 + 2 + 1 + 8 + _LBP_SIZE + _HOG_SIZE + _HSV_SIZE   # = 3908

# ── Helpers ───────────────────────────────────────────────────────
def _g(f):  return cv.cvtColor(f, cv.COLOR_BGR2GRAY) if f.ndim==3 else f
def _cl(g): return cv.createCLAHE(3.0,(8,8)).apply(g)
def _st(s): return "Verdadeira" if s>=62 else ("Falsa" if s<=38 else "Incerta")
def _r(st, sc, mt, **kw): return {"status":st,"score":float(np.clip(sc,0,100)),"metodo":mt,**kw}

# ── 1. GRAY_STATS ─────────────────────────────────────────────────
def por_gray(f):
    g = _g(f).astype(np.float32)
    d = float(np.std(g))
    e = float(shannon_entropy(g/255)) if SKIMAGE_OK else 5.0
    s = .5*np.clip((d-20)/60*100,0,100) + .5*np.clip((e-3)/4.5*100,0,100)
    return _r(_st(s),s,"GRAY", desvio=d, entropia=e)

# ── 2. SOBEL ──────────────────────────────────────────────────────
def por_sobel(f):
    g  = _cl(_g(f))
    sx = cv.Sobel(g,cv.CV_64F,1,0,ksize=3); sy = cv.Sobel(g,cv.CV_64F,0,1,ksize=3)
    mag = np.sqrt(sx**2+sy**2)
    d, u = float(mag.mean()), float(1-mag.std()/(mag.mean()+1e-6))
    s = .6*np.clip((d-10)/80*100,0,100) + .4*np.clip(u*100,0,100)
    return _r(_st(s),s,"SOBEL", densidade=d, uniformidade=u)

# ── 3. LAPLACE ────────────────────────────────────────────────────
def por_laplace(f):
    v = float(cv.Laplacian(_g(f),cv.CV_64F).var())
    s = np.clip((v-50)/950*100,0,100)
    return _r(_st(s),s,"LAPLACE", variancia_lap=v)

# ── 4. FFT ────────────────────────────────────────────────────────
def por_fft(f):
    g   = _g(f).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h,w = mag.shape; cy,cx = h//2,w//2; rm = min(cy,cx)
    Y,X = np.ogrid[:h,:w]; d = np.sqrt((X-cx)**2+(Y-cy)**2)
    anel = (d>rm*.10)&(d<rm*.40)
    pot  = float(mag[anel].mean())
    n    = int(anel.sum() * 0.05)
    s    = .5*np.clip(pot/5000*100,0,100) + .5*np.clip(n/800*100,0,100)
    return _r(_st(s),s,"FFT", pot_media_alta=pot, n_picos=n)

# ── 5. LBP ────────────────────────────────────────────────────────
def por_lbp(f):
    if not SKIMAGE_OK: return _r("Incerta",50,"LBP",erro="scikit-image ausente")
    lbp = local_binary_pattern(_g(f), P=16, R=2, method="uniform")
    h,_ = np.histogram(lbp.ravel(), bins=18, range=(0,18), density=True)
    e   = float(shannon_entropy(h+1e-10))
    s   = np.clip((e-1.5)/1.5*100,0,100)
    return _r(_st(s),s,"LBP", entropia_lbp=e)

# ── 6. HIST_COR ───────────────────────────────────────────────────
_FAIXAS = [(100,130,80,255,80,255),(0,15,80,255,80,255),(20,35,80,255,80,255)]

def por_hist_cor(f):
    hsv = cv.cvtColor(f,cv.COLOR_BGR2HSV); tot = f.shape[0]*f.shape[1]
    sf  = [float(np.clip(cv.inRange(hsv,(h0,s0,v0),(h1,s1,v1)).sum()/tot*500,0,100))
           for h0,h1,s0,s1,v0,v1 in _FAIXAS]
    s   = float(np.mean(sf))
    return _r(_st(s),s,"HIST_COR", scores_faixas=sf)

# ── 7. ORB ────────────────────────────────────────────────────────
def salvar_template_orb(frame):
    orb = cv.ORB_create(1000)
    kps, des = orb.detectAndCompute(_g(frame), None)
    with open(PATH_TPL,"wb") as f:
        pickle.dump({"kps":[(k.pt[0],k.pt[1],k.size,k.angle,k.response,k.octave,k.class_id)
                             for k in kps], "des": des.tolist() if des is not None else []}, f)
    log.info("Template ORB salvo: %d keypoints.", len(kps))

def por_orb(f):
    if not os.path.exists(PATH_TPL): return _r("Incerta",50,"ORB",aviso="sem template")
    with open(PATH_TPL,"rb") as fp: data = pickle.load(fp)
    des_r = np.array(data["des"], dtype=np.uint8)
    if not len(des_r): return _r("Incerta",50,"ORB",aviso="template vazio")
    orb = cv.ORB_create(1000)
    _, des = orb.detectAndCompute(_g(f), None)
    if des is None: return _r("Falsa",10,"ORB",matches=0)
    bons = [m for m in cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True).match(des,des_r) if m.distance<60]
    s = np.clip(len(bons)/80*100,0,100)
    return _r(_st(s),s,"ORB",matches=len(bons))

# ── 8. HOG ────────────────────────────────────────────────────────
_HOG_P = dict(orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm="L2-Hys")

def _hog_feat(f) -> np.ndarray:
    """
    Retorna vetor HOG de tamanho FIXO (_HOG_SIZE=3780).
    Se skimage não estiver disponível, retorna zeros do mesmo tamanho.
    """
    if not SKIMAGE_OK:
        return np.zeros(_HOG_SIZE, dtype=np.float32)
    feat = sk_hog(cv.resize(_g(f),(256,128)), feature_vector=True, **_HOG_P)
    feat = feat.astype(np.float32)
    # Garante tamanho exato mesmo que parâmetros mudem futuramente
    if feat.shape[0] != _HOG_SIZE:
        out = np.zeros(_HOG_SIZE, dtype=np.float32)
        n   = min(feat.shape[0], _HOG_SIZE)
        out[:n] = feat[:n]
        return out
    return feat

def por_hog(f, modelo=None):
    """
    Avalia a energia do vetor HOG heuristicamente.
    NUNCA passa _hog_feat ao modelo SVM diretamente — o SVM foi treinado
    com _features() (3908 dims), não com _hog_feat() (3780 dims).
    Para classificação via SVM use TreinadorNotas.classificar().
    """
    if not SKIMAGE_OK: return _r("Incerta",50,"HOG",erro="skimage ausente")
    feat = _hog_feat(f)
    s = np.clip((np.linalg.norm(feat)-5)/25*100,0,100)
    return _r(_st(s),s,"HOG",energia=float(np.linalg.norm(feat)))

# ── 9. FACE ───────────────────────────────────────────────────────
def por_face(f):
    if not FACE_OK: return _r("Incerta",50,"FACE",erro="face_recognition ausente")
    locs = face_recognition.face_locations(cv.cvtColor(f,cv.COLOR_BGR2RGB),model="hog")
    if not locs: return _r("Incerta",50,"FACE",faces=0)
    s = min(70+len(locs)*10,90)
    return _r("Verdadeira",s,"FACE",faces=len(locs))

# ── 10. COMBINADO ─────────────────────────────────────────────────
def por_combinado(f, modelo=None):
    subs = {
        "GRAY":por_gray(f),"SOBEL":por_sobel(f),"LAPLACE":por_laplace(f),
        "FFT":por_fft(f),"LBP":por_lbp(f),"HIST_COR":por_hist_cor(f),
        "ORB":por_orb(f),"HOG":por_hog(f),"FACE":por_face(f),
    }
    sc_t = pw_t = 0.0
    for k,p in _PESOS.items():
        if subs[k]["status"]!="Incerta": sc_t+=subs[k]["score"]*p; pw_t+=p
    sc    = float(np.clip(sc_t/pw_t if pw_t else 50,0,100))
    votos = {s:sum(_PESOS[k] for k in _PESOS if subs[k]["status"]==s)
             for s in ("Verdadeira","Falsa","Incerta")}
    return _r(max(votos,key=votos.get),sc,"COMBINADO",detalhes=subs,votos=votos)

# ══════════════════════════════════════════════════════════════════
# VETOR DE FEATURES (tamanho fixo = _FEAT_SIZE = 3908)
# ══════════════════════════════════════════════════════════════════

def _features(f) -> np.ndarray:
    """
    Extrai vetor de features de tamanho FIXO (_FEAT_SIZE).
    Blocos:
        [0:3]           – gray stats  (média, desvio, entropia)
        [3:5]           – sobel       (densidade, desvio)
        [5:6]           – laplace     (variância)
        [6:14]          – fft         (8 anéis de frequência)
        [14:32]         – lbp         (18 bins, zeros se skimage ausente)
        [32:3812]       – hog         (3780 valores, zeros se skimage ausente)
        [3812:3908]     – hsv hist    (96 bins)
    """
    g   = _g(f)
    cl  = _cl(g)

    # Gray stats
    e      = float(shannon_entropy(g/255)) if SKIMAGE_OK else 0.0
    f_gray = np.array([float(np.mean(g)), float(np.std(g)), e], dtype=np.float32)

    # Sobel
    sx  = cv.Sobel(cl,cv.CV_64F,1,0,ksize=3)
    sy  = cv.Sobel(cl,cv.CV_64F,0,1,ksize=3)
    mag = np.sqrt(sx**2+sy**2)
    f_sob = np.array([mag.mean(), mag.std()], dtype=np.float32)

    # Laplace
    f_lap = np.array([cv.Laplacian(g,cv.CV_64F).var()], dtype=np.float32)

    # FFT – 8 anéis de potência espectral
    fft  = np.abs(np.fft.fftshift(np.fft.fft2(g.astype(np.float32))))
    h,w  = fft.shape; cy,cx = h//2,w//2; rm = min(cy,cx)
    Y,X  = np.ogrid[:h,:w]; dist = np.sqrt((X-cx)**2+(Y-cy)**2)
    f_fft = np.array([fft[(dist>=rm*i/8)&(dist<rm*(i+1)/8)].mean()
                      for i in range(8)], dtype=np.float32)

    # LBP – sempre 18 bins (zeros se skimage ausente)
    f_lbp = np.zeros(_LBP_SIZE, dtype=np.float32)
    if SKIMAGE_OK:
        lbp      = local_binary_pattern(g, P=16, R=2, method="uniform")
        hist, _  = np.histogram(lbp.ravel(), bins=18, range=(0,18), density=True)
        f_lbp[:] = hist.astype(np.float32)

    # HOG – sempre 3780 (zeros se skimage ausente)
    f_hog = _hog_feat(f)   # já retorna array de tamanho fixo _HOG_SIZE

    # HSV histograma – sempre 96 bins (3 canais × 32)
    hsv   = cv.cvtColor(f, cv.COLOR_BGR2HSV)
    f_hsv = np.concatenate([
        cv.calcHist([hsv],[i],None,[32],[0,256]).flatten() / (g.size+1e-6)
        for i in range(3)
    ]).astype(np.float32)

    vetor = np.concatenate([f_gray, f_sob, f_lap, f_fft, f_lbp, f_hog, f_hsv])

    # Verificação de sanidade — corrige silenciosamente em vez de travar
    if vetor.shape[0] != _FEAT_SIZE:
        log.error("Tamanho do vetor %d != %d esperado. Ajustando.",
                  vetor.shape[0], _FEAT_SIZE)
        out = np.zeros(_FEAT_SIZE, dtype=np.float32)
        n   = min(vetor.shape[0], _FEAT_SIZE)
        out[:n] = vetor[:n]
        return out

    return vetor


# ══════════════════════════════════════════════════════════════════
# TREINADOR SVM
# ══════════════════════════════════════════════════════════════════

class TreinadorNotas:
    """Coleta amostras, treina SVM e classifica cédulas BRL."""

    def __init__(self):
        self.X: List[np.ndarray] = []
        self.y: List[int]        = []
        self.modelo = None
        self._carregar()

    def _carregar(self):
        """Carrega modelo salvo, descartando se incompatível com _FEAT_SIZE."""
        if not (SKLEARN_OK and os.path.exists(PATH_SVM)):
            return
        try:
            m = joblib.load(PATH_SVM)
            # Verifica se o scaler foi treinado com o mesmo número de features
            n_feats = m.named_steps["sc"].n_features_in_
            if n_feats != _FEAT_SIZE:
                log.warning(
                    "Modelo descartado: treinado com %d features, atual=%d. "
                    "Delete '%s' e retreine.", n_feats, _FEAT_SIZE, PATH_SVM
                )
                os.remove(PATH_SVM)
                return
            self.modelo = m
            log.info("Modelo SVM carregado (%d features).", _FEAT_SIZE)
        except Exception as e:
            log.warning("Falha ao carregar SVM: %s", e)

    def adicionar(self, frame: np.ndarray, verdadeira: bool) -> None:
        """Extrai features do frame e armazena a amostra rotulada."""
        self.X.append(_features(frame))
        self.y.append(int(verdadeira))
        v = len([i for i in self.y if i==1])
        f = len([i for i in self.y if i==0])
        log.info("Amostra %s adicionada. Total=%d (V=%d F=%d)",
                 "Verdadeira" if verdadeira else "Falsa", len(self.y), v, f)

    def adicionar_de_pasta(self, pasta_v: str, pasta_f: str) -> int:
        n = 0
        for pasta, rot in ((pasta_v,True),(pasta_f,False)):
            for nome in sorted(os.listdir(pasta)):
                if not nome.lower().endswith((".jpg",".jpeg",".png",".bmp")): continue
                img = cv.imread(os.path.join(pasta,nome))
                if img is None: continue
                self.adicionar(img,rot); n+=1
        return n

    def treinar(self, cv5=True) -> float:
        """Treina o SVM. Retorna acurácia cross-val (0.0 se amostras < 10)."""
        if not SKLEARN_OK: raise ImportError("scikit-learn necessário.")
        if len(self.X) < 4:
            raise ValueError(f"Mínimo 4 amostras (atual: {len(self.X)}). "
                             "Pressione v/f para coletar mais.")
        X = np.array(self.X, dtype=np.float32)
        y = np.array(self.y, dtype=int)

        # Garante pelo menos 1 amostra de cada classe
        if len(set(y)) < 2:
            raise ValueError("Precisa de amostras de AMBAS as classes (v e f).")

        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, random_state=42)),
        ])

        acc = 0.0
        if cv5 and len(X) >= 10:
            folds = min(5, min(np.bincount(y)))   # evita fold > menor classe
            if folds >= 2:
                acc = float(cross_val_score(pipe, X, y, cv=folds,
                                            scoring="accuracy").mean())
                log.info("Cross-val (%d-fold): %.1f%%", folds, acc*100)

        pipe.fit(X, y)
        self.modelo = pipe
        joblib.dump(pipe, PATH_SVM)
        log.info("SVM salvo em '%s'. Features=%d", PATH_SVM, _FEAT_SIZE)
        return acc

    def classificar(self, frame: np.ndarray) -> dict:
        """
        Classifica com o modelo SVM usando o vetor completo _features() (3908 dims).
        Fallback para COMBINADO se modelo ausente ou incompatível.
        """
        if self.modelo is None:
            log.debug("SVM indisponível, usando COMBINADO.")
            return por_combinado(frame)
        try:
            feat = _features(frame).reshape(1, -1)
            # Verificação de compatibilidade antes de chamar predict_proba
            n_esperado = self.modelo.named_steps["sc"].n_features_in_
            if feat.shape[1] != n_esperado:
                log.warning(
                    "Incompatibilidade de features: gerou %d, modelo espera %d. "
                    "Descartando modelo. Retreine com 't'.",
                    feat.shape[1], n_esperado
                )
                self.modelo = None
                if os.path.exists(PATH_SVM):
                    os.remove(PATH_SVM)
                return por_combinado(frame)
            p = self.modelo.predict_proba(feat)[0]
            s = float(p[1]) * 100
            return _r(_st(s), s, "SVM", prob_v=float(p[1]), prob_f=float(p[0]))
        except ValueError as e:
            log.error("Erro no SVM (%s). Usando COMBINADO.", e)
            self.modelo = None
            return por_combinado(frame)

    @property
    def n_amostras(self): return len(self.y)

    @property
    def n_verdadeiras(self): return sum(self.y)

    @property
    def n_falsas(self): return len(self.y) - sum(self.y)


# ── Instância global ──────────────────────────────────────────────
_treinador = TreinadorNotas()

def classificar(frame: np.ndarray, modo: str = "COMBINADO") -> dict:
    m = modo.upper()
    return {
        "GRAY":     lambda: por_gray(frame),
        "SOBEL":    lambda: por_sobel(frame),
        "LAPLACE":  lambda: por_laplace(frame),
        "FFT":      lambda: por_fft(frame),
        "LBP":      lambda: por_lbp(frame),
        "HIST_COR": lambda: por_hist_cor(frame),
        "ORB":      lambda: por_orb(frame),
        "HOG":      lambda: por_hog(frame),          # sem modelo — usa heurística
        "FACE":     lambda: por_face(frame),
        "SVM":      lambda: _treinador.classificar(frame),   # único que usa SVM
        "COMBINADO":lambda: por_combinado(frame),
    }.get(m, lambda: por_combinado(frame))()

def obter_treinador() -> TreinadorNotas: return _treinador


# ── CLI de treinamento ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Treinamento offline do classificador BRL.")
    p.add_argument("--treinar",     action="store_true")
    p.add_argument("--verdadeiras", default=os.path.join(_DIR,"dataset","verdadeiras"))
    p.add_argument("--falsas",      default=os.path.join(_DIR,"dataset","falsas"))
    p.add_argument("--template",    default=None, help="Imagem para template ORB")
    a = p.parse_args()

    if a.template:
        img = cv.imread(a.template)
        if img is not None: salvar_template_orb(img); print("Template ORB salvo.")
        else: print("Imagem não encontrada.")

    if a.treinar:
        if not (os.path.isdir(a.verdadeiras) and os.path.isdir(a.falsas)):
            print(f"Crie as pastas:\n  {a.verdadeiras}\n  {a.falsas}")
        else:
            t = TreinadorNotas()
            n = t.adicionar_de_pasta(a.verdadeiras, a.falsas)
            if n:
                acc = t.treinar()
                print(f"\nAmostras: {t.n_verdadeiras} verdadeiras + {t.n_falsas} falsas")
                print(f"Acurácia CV: {acc*100:.1f}%  |  Modelo: {PATH_SVM}")
            else:
                print("Nenhuma imagem encontrada nas pastas.")
