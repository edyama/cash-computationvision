"""testes.py – Testes unitários do detector de notas BRL.  Execute: python testes.py -v"""

import os
import sys
import json
import unittest
import numpy as np
import cv2 as cv
sys.path.insert(0, os.path.dirname(__file__))
import preprocessamento as proc, classificador as classf, comunicacao as com


def _nota(rica=True, w=640, h=300):
    """Frame sintético: rica=True → imita nota verdadeira, False → falsa."""
    img = np.zeros((h,w,3),dtype=np.uint8)
    if rica:
        img = np.random.randint(60,200,(h,w,3),dtype=np.uint8)
        for i in range(0,h,4): cv.line(img,(0,i),(w,i+2),(180,140,80),1)
        cv.rectangle(img,(10,10),(w-10,h-10),(50,80,30),2)
    else:
        img[:] = (120,150,100); cv.GaussianBlur(img,(15,15),0,img)
    return img

NV, NF = _nota(True), _nota(False)
_RES   = {"status":"Verdadeira","score":88.,"metodo":"TEST",
          "detalhes":{"GRAY":{"status":"Verdadeira","score":80},
                      "SOBEL":{"status":"Verdadeira","score":75}}}


# ── Pré-processamento ─────────────────────────────────────────────
class TestPreproc(unittest.TestCase):
    def test_chaves(self):
        r = proc.preprocessar(NV)
        for k in ("original","gray","clahe","sobel","laplace","canny","thresh","fft_mag","nitidez"):
            self.assertIn(k, r)
    def test_largura_normalizada(self):
        self.assertEqual(proc.preprocessar(np.zeros((600,1280,3),dtype=np.uint8))["original"].shape[1], proc.W)
    def test_none_erro(self):
        with self.assertRaises(Exception): proc.preprocessar(None)
    def test_marcar_shape(self):
        self.assertEqual(proc.marcar(NV,_RES).shape, NV.shape)
    def test_painel_3canais(self):
        self.assertEqual(len(proc.painel(proc.preprocessar(NV)).shape), 3)


# ── Técnicas ──────────────────────────────────────────────────────
class TestTecnicas(unittest.TestCase):

    def _check(self, r):
        self.assertIn(r["status"],("Verdadeira","Falsa","Incerta"))
        self.assertGreaterEqual(r["score"],0); self.assertLessEqual(r["score"],100)

    def test_gray_score(self):     [self._check(classf.por_gray(f))    for f in (NV,NF)]
    def test_sobel_score(self):    [self._check(classf.por_sobel(f))   for f in (NV,NF)]
    def test_laplace_score(self):  [self._check(classf.por_laplace(f)) for f in (NV,NF)]
    def test_fft_score(self):      [self._check(classf.por_fft(f))     for f in (NV,NF)]
    def test_lbp_score(self):      [self._check(classf.por_lbp(f))     for f in (NV,NF)]
    def test_hist_cor_score(self): [self._check(classf.por_hist_cor(f))for f in (NV,NF)]
    def test_orb_sem_template(self):
        orig = classf.PATH_TPL; classf.PATH_TPL = "/tmp/_nao_existe.pkl"
        self._check(classf.por_orb(NV)); classf.PATH_TPL = orig
    def test_hog_score(self):      [self._check(classf.por_hog(f))     for f in (NV,NF)]
    def test_face_score(self):     [self._check(classf.por_face(f))    for f in (NV,NF)]

    def test_gray_verdadeira_maior(self):
        self.assertGreater(classf.por_gray(NV)["score"], classf.por_gray(NF)["score"])
    def test_laplace_verdadeira_maior(self):
        self.assertGreater(classf.por_laplace(NV)["variancia_lap"],
                           classf.por_laplace(NF)["variancia_lap"])
    def test_sobel_densidade_maior(self):
        self.assertGreater(classf.por_sobel(NV)["densidade"],
                           classf.por_sobel(NF)["densidade"])
    def test_lbp_entropia_maior(self):
        if not classf.SKIMAGE_OK: self.skipTest("skimage ausente")
        self.assertGreater(classf.por_lbp(NV).get("entropia_lbp",0),
                           classf.por_lbp(NF).get("entropia_lbp",0))


# ── Combinado ─────────────────────────────────────────────────────
class TestCombinado(unittest.TestCase):
    def test_detalhes(self):
        r = classf.por_combinado(NV)
        for m in ("GRAY","SOBEL","LAPLACE","FFT","LBP","HIST_COR","ORB","HOG","FACE"):
            self.assertIn(m, r["detalhes"])
    def test_votos_soma_1(self):
        self.assertAlmostEqual(sum(classf.por_combinado(NV)["votos"].values()),1.,places=5)
    def test_score_0_100(self):
        for f in (NV,NF):
            r = classf.por_combinado(f)
            self.assertGreaterEqual(r["score"],0); self.assertLessEqual(r["score"],100)


# ── Interface unificada ───────────────────────────────────────────
class TestClassificar(unittest.TestCase):
    def _ok(self, m):
        r = classf.classificar(NV, modo=m)
        self.assertIn("status",r); self.assertIn("score",r)
    def test_todos_modos(self):
        for m in classf.MODOS: self._ok(m)


# ── Treinador ─────────────────────────────────────────────────────
class TestTreinador(unittest.TestCase):
    def test_adicionar(self):
        t = classf.TreinadorNotas()
        t.adicionar(NV,True); t.adicionar(NF,False)
        self.assertEqual(len(t.X),2); self.assertEqual(t.y,[1,0])
    def test_treinar_minimo(self):
        t = classf.TreinadorNotas(); t.adicionar(NV,True)
        with self.assertRaises(Exception): t.treinar()
    def test_treinar_e_classificar(self):
        if not classf.SKLEARN_OK: self.skipTest("scikit-learn ausente")
        t = classf.TreinadorNotas()
        for _ in range(6): t.adicionar(NV,True); t.adicionar(NF,False)
        t.treinar(cv5=False)
        r = t.classificar(NV)
        self.assertIn(r["status"],("Verdadeira","Falsa","Incerta"))
    def test_features_shape_consistente(self):
        self.assertEqual(classf._features(NV).shape, classf._features(NF).shape)


# ── Comunicação ───────────────────────────────────────────────────
class TestComun(unittest.TestCase):
    CSV, JSN = "/tmp/_t_notas.csv", "/tmp/_t_notas.json"
    def setUp(self):
        for p in (self.CSV,self.JSN):
            if os.path.exists(p): os.remove(p)
    def test_csv_cabecalho(self):
        com.registrar_csv(self.CSV,"c0",_RES)
        self.assertIn("status", open(self.CSV).readline())
    def test_csv_2linhas(self):
        com.registrar_csv(self.CSV,"c0",_RES)
        self.assertEqual(len(open(self.CSV).readlines()),2)
    def test_json_campos(self):
        com.salvar_json(self.JSN,"c0",_RES)
        d=json.load(open(self.JSN)); self.assertIn("status",d)
    def test_stats(self):
        s=com.Estatisticas(); s.atualizar(_RES); self.assertEqual(s.d["Verdadeira"],1)
    def test_resetar(self):
        s=com.Estatisticas(); s.atualizar(_RES); s.resetar()
        self.assertEqual(s.d["Verdadeira"],0); self.assertEqual(s.n,0)
    def test_exibir_nao_falha(self):
        com.exibir("c0",_RES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
