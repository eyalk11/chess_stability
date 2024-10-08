import chess
import chess.pgn
# import numpy as np
import io
import sys
from sunfish.tools import uci
from sunfish.tools.uci import *
from sunfish import sunfish
import timeit

if __name__ == '__main__':
    uci.sunfish=sunfish

    # fen='r2Rr1k1/pp3pp1/2p2bbp/4n3/4pBP1/2Q1P1NP/PPP2P2/1K3B1R b - - 0 17'
    fen=    'r2q1rk1/3nbppp/b1pp1n2/p3p3/1p1PP1P1/P5NP/1PP1NPB1/R1BQ1RK1 b - - 1 12' 
    if sys.argv[1]!='aa':
        from calc import Calculator
        if sys.platform=='win32':
            STOCKFISHPATH=r'c:/gitproj/stockfish/stockfish-windows-x86-64-avx2.exe'
        else:
            STOCKFISHPATH='/home/jovyan/stockfish/src/stockfish'

        def my_function(): 
            mycalc=Calculator.from_engine_path(STOCKFISHPATH)
            b=chess.Board(fen)
            mycalc.print_stats(b,0)
            mycalc.printtimer() 
            mycalc.end()
        start_time = timeit.default_timer()
        my_function()
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        print(f"The function took {execution_time} seconds to execute")




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
