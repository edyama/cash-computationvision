"""preprocessamento.py – Camadas de análise para cédulas BRL."""

import cv2 as cv
import numpy as np

W, H = 640, 300
COR  = {"Verdadeira": (0,200,0), "Falsa": (0,0,220), "Incerta": (0,200,255)}


def preprocessar(frame: np.ndarray) -> dict:
    """Retorna dict com 9 camadas de análise da imagem."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame inválido.")
    orig = cv.resize(frame, (W, int(frame.shape[0] * W / frame.shape[1])))
    g    = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    cl   = cv.createCLAHE(3.0, (8,8)).apply(g)
    sx   = cv.Sobel(cl, cv.CV_64F, 1, 0, ksize=3)
    sy   = cv.Sobel(cl, cv.CV_64F, 0, 1, ksize=3)
    fft  = np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    m    = cv.blur(g.astype(np.float32), (7,7))
    return {
        "original": orig,
        "gray":     g,
        "clahe":    cl,
        "sobel":    np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255)),
        "laplace":  np.uint8(np.clip(np.abs(cv.Laplacian(cl, cv.CV_64F)), 0, 255)),
        "canny":    cv.Canny(cl, 50, 150),
        "thresh":   cv.adaptiveThreshold(cl,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,4),
        "fft_mag":  np.uint8(np.clip(20*np.log(np.abs(fft)+1)/np.log(fft.size+1)*12,0,255)),
        "nitidez":  np.uint8(np.clip(cv.blur(g.astype(np.float32)**2,(7,7)) - m**2,0,255)),
        "hsv":      cv.cvtColor(orig, cv.COLOR_BGR2HSV),
        "lab":      cv.cvtColor(orig, cv.COLOR_BGR2Lab),
    }


def marcar(frame: np.ndarray, r: dict) -> np.ndarray:
    """Sobrepõe veredito e scores na imagem."""
    out = frame.copy()
    st, sc, mt = r.get("status","Incerta"), r.get("score",50.0), r.get("metodo","-")
    cor = COR.get(st, (200,200,200))
    h, w = out.shape[:2]
    cv.rectangle(out, (0,0), (w-1,h-1), cor, 4)
    ov = out.copy(); cv.rectangle(ov,(0,0),(w,52),(0,0,0),-1)
    cv.addWeighted(ov,.55,out,.45,0,out)
    cv.putText(out, f"{st}  {sc:.1f}%", (10,22), cv.FONT_HERSHEY_DUPLEX, .75, cor, 2)
    cv.putText(out, mt, (10,44), cv.FONT_HERSHEY_SIMPLEX, .45, (200,200,200), 1)
    x = 10
    for nome, sub in r.get("detalhes",{}).items():
        if x > w-100: break
        txt = f"{nome}:{sub.get('score',0):.0f}%"
        cv.putText(out, txt, (x, h-8), cv.FONT_HERSHEY_SIMPLEX, .38,
                    COR.get(sub.get("status","Incerta"),(180,180,180)), 1)
        x += len(txt)*8+4
    return out


def painel(frames: dict) -> np.ndarray:
    """Grade 3×3 com as camadas de pré-processamento."""
    th, tw = H//3, W//3
    tiles = []
    for k in ["gray","clahe","sobel","laplace","canny","thresh","fft_mag","nitidez","original"]:
        img = cv.resize(frames.get(k, np.zeros((th,tw),dtype=np.uint8)), (tw,th))
        if img.ndim == 2: img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.putText(img, k, (4,14), cv.FONT_HERSHEY_SIMPLEX, .4, (0,255,255), 1)
        tiles.append(img)
    return np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,9,3)])
