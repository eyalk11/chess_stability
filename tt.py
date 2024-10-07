import chess
import ipywidgets as widgets
import chess.pgn
import numpy as np
import io
import sys
from sunfish.tools import uci
from sunfish.tools.uci import *
from sunfish import sunfish

uci.sunfish=sunfish

fen='r2Rr1k1/pp3pp1/2p2bbp/4n3/4pBP1/2Q1P1NP/PPP2P2/1K3B1R b - - 0 17'
if sys.argv[1]!='aa':
    from calc import Calculator
    if sys.platform=='win32':
        STOCKFISHPATH=r'c:/gitproj/stockfish/stockfish-windows-x86-64-avx2.exe'
    else:
        STOCKFISHPATH='/home/jovyan/stockfish/src/stockfish'

    mycalc=Calculator.from_engine_path(STOCKFISHPATH)
    b=chess.Board(fen)
    mycalc.print_stats(b,0)


else:
#sys.path+=['c:\\gitproj\\sunfish2\\src']
#    from tools import uci
#    from tools.uci import *
#    import sunfish





    pos=uci.from_fen(*fen.split(" "))
    ##




    hist = [pos] if uci.get_color(pos) == WHITE else [pos.rotate(), pos]

    searcher = sunfish.Searcher()

    print(searcher.bound(hist[-1], 0, 5, can_null=False))
    print(searcher.nodes)
