"""comunicacao.py – Console · CSV · JSON · Estatísticas para detector BRL."""

import csv
import json
import os
import time
from datetime import datetime

_C = {"Verdadeira":"\033[92m","Falsa":"\033[91m","Incerta":"\033[93m"}
_R = "\033[0m"; _B = "\033[1m"
_I = {"Verdadeira":"✔ VERDADEIRA","Falsa":"✘ FALSA","Incerta":"? INCERTA"}


def exibir(cam, r: dict) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st, sc, mt = r.get("status","Incerta"), r.get("score",50.), r.get("metodo","-")
    print(f"{_B}[{ts}]{_R} cam={cam}  {_C.get(st,'')}{_B}{_I.get(st,'')}{_R}  {sc:.1f}%  [{mt}]")
    for k, sub in r.get("detalhes",{}).items():
        print(f"          {_C.get(sub.get('status',''),'')}{k:<10}{_R} {sub.get('score',0):5.1f}%")


def instrucoes() -> None:
    print(f"\n{_B}{'─'*50}{_R}")
    print("  q=sair  m=modo  s=screenshot  d=diagnóstico")
    print("  v=amostra Verdadeira  f=amostra Falsa  t=treinar")
    print(f"{_B}{'─'*50}{_R}\n")


def registrar_csv(path, cam, r: dict) -> None:
    novo = not os.path.exists(path)
    with open(path,"a",newline="") as f:
        w = csv.writer(f)
        if novo: w.writerow(["timestamp","camera","metodo","status","score","detalhes"])
        det = json.dumps({k:v.get("score") for k,v in r.get("detalhes",{}).items()})
        w.writerow([datetime.now().isoformat(timespec="milliseconds"),
                    cam, r.get("metodo",""), r.get("status",""),
                    f"{r.get('score',0):.2f}", det])


def salvar_json(path, cam, r: dict) -> None:
    with open(path,"w") as f:
        json.dump({"timestamp":datetime.now().isoformat(),"camera":str(cam),
                   "status":r.get("status"),"score":round(r.get("score",0),2),
                   "metodo":r.get("metodo"),
                   "detalhes":{k:{"score":v.get("score"),"status":v.get("status")}
                               for k,v in r.get("detalhes",{}).items()}},
                  f, ensure_ascii=False, indent=2)


class Estatisticas:
    def __init__(self):
        self.d={"Verdadeira":0,"Falsa":0,"Incerta":0}; self.n=0; self._t=time.time()
    def atualizar(self, r):
        self.n+=1; self.d[r.get("status","Incerta")]=self.d.get(r.get("status","Incerta"),0)+1
    def exibir(self):
        fps=self.n/max(time.time()-self._t,1e-6); tot=sum(self.d.values()) or 1
        print(f"\n{_B}── FPS={fps:.1f}  Frames={self.n} ──{_R}")
        for st,n in self.d.items():
            print(f"  {_C.get(st,'')}{st:<12}{_R} {n:4d}  ({n/tot*100:.1f}%)")
    def resetar(self):
        self.d={"Verdadeira":0,"Falsa":0,"Incerta":0}; self.n=0; self._t=time.time()


def alertar_falsa():
    print(f"\a{_C['Falsa']}{_B}⚠  NOTA FALSA DETECTADA  ⚠{_R}\n")
