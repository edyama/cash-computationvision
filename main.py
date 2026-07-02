"""
main.py – Detector de autenticidade de cédulas BRL.

  python main.py                          # câmera 0, modo COMBINADO
  python main.py --modo SOBEL --cameras 0 1
  python main.py --imagem nota.jpg        # imagem estática
  python main.py --csv log.csv --diagnostico

Teclas: q=sair  m=modo  s=screenshot  d=diagnóstico
        v=amostra Verdadeira  f=amostra Falsa  t=treinar SVM
"""

import argparse
import logging
import threading
import time
import cv2 as cv
import preprocessamento as proc
import classificador    as classf
import comunicacao      as com

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("main")


def loop_camera(cam_id, modo_ref, args, stats, treinador):
    cap = cv.VideoCapture(cam_id)
    if not cap.isOpened():
        log.error("Câmera %d indisponível.", cam_id); return

    janela  = f"Detector BRL – Cam {cam_id}"
    diag    = [args.diagnostico]
    ultimo  = [None]
    shot    = [0]
    com.instrucoes()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frames  = proc.preprocessar(frame)
        orig    = frames["original"]
        ultimo[0] = orig.copy()

        r       = classf.classificar(orig, modo=modo_ref[0])
        anotado = proc.marcar(orig, r)
        cv.imshow(janela, anotado)
        if diag[0]: cv.imshow("Diagnóstico", proc.painel(frames))

        com.exibir(cam_id, r)
        if args.csv:  com.registrar_csv(args.csv, cam_id, r)
        if args.json: com.salvar_json(args.json, cam_id, r)
        if r["status"] == "Falsa": com.alertar_falsa()
        stats.atualizar(r)

        k = cv.waitKey(1) & 0xFF
        if   k == ord("q"): break
        elif k == ord("m"):
            modo_ref[0] = classf.MODOS[(classf.MODOS.index(modo_ref[0])+1)%len(classf.MODOS)]
            log.info("Modo → %s", modo_ref[0])
        elif k == ord("s"):
            arq=f"shot_{cam_id}_{shot[0]}.jpg"; cv.imwrite(arq,anotado); shot[0]+=1
            log.info("Screenshot: %s", arq)
        elif k == ord("d"):
            diag[0] = not diag[0]
            if not diag[0]: cv.destroyWindow("Diagnóstico")
        elif k == ord("v") and ultimo[0] is not None:
            treinador.adicionar(ultimo[0], True)
        elif k == ord("f") and ultimo[0] is not None:
            treinador.adicionar(ultimo[0], False)
        elif k == ord("t"):
            try:    log.info("Acurácia CV: %.1f%%", treinador.treinar()*100)
            except Exception as e: log.error("Erro treino: %s", e)
        elif k == ord("r"):
            stats.resetar()

    cap.release(); cv.destroyAllWindows()


def processar_imagem(args, treinador):
    frame = cv.imread(args.imagem)
    if frame is None:
        log.error("Imagem não encontrada: %s", args.imagem); return
    frames = proc.preprocessar(frame)
    r      = classf.classificar(frames["original"], modo=args.modo)
    com.exibir("imagem", r)
    if args.csv:  com.registrar_csv(args.csv, "imagem", r)
    if args.json: com.salvar_json(args.json, "imagem", r)
    cv.imshow(f"[{args.modo}] q=sair  d=diagnóstico", proc.marcar(frames["original"],r))
    while True:
        k = cv.waitKey(0) & 0xFF
        if k in (ord("q"),27): break
        elif k == ord("d"): cv.imshow("Diagnóstico", proc.painel(frames))
        elif k == ord("s"):
            arq=args.imagem.replace(".",f"_{args.modo}."); cv.imwrite(arq,proc.marcar(frames["original"],r))
    cv.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Detector de autenticidade de cédulas BRL.")
    p.add_argument("--cameras",    nargs="+", type=int, default=[0])
    p.add_argument("--imagem",     type=str,  default=None)
    p.add_argument("--modo",       type=str,  default="COMBINADO", choices=classf.MODOS)
    p.add_argument("--csv",        type=str,  default=None)
    p.add_argument("--json",       type=str,  default=None)
    p.add_argument("--diagnostico",action="store_true")
    p.add_argument("--stats-seg",  type=int,  default=30)
    args = p.parse_args()

    treinador = classf.obter_treinador()
    stats     = com.Estatisticas()

    if args.imagem:
        processar_imagem(args, treinador); return

    threads = [threading.Thread(target=loop_camera,
                                args=(c,[args.modo],args,stats,treinador),
                                daemon=True) for c in args.cameras]
    for t in threads: t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(args.stats_seg); stats.exibir()
    except KeyboardInterrupt: pass
    for t in threads: t.join(timeout=3)
    stats.exibir()


if __name__ == "__main__":
    main()
