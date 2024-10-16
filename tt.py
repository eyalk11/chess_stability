import asyncio

import chess
import chess.pgn
try:
    import numpy as np
except:
    pass
import io
import sys
from sunfish.tools import uci
from sunfish.tools.uci import *
from sunfish import sunfish
import timeit


pgn= '''[Event "2021-tata-steel-masters"]
[Site "?"]
[Date "Sun Jan 31 2021"]
[Round "13"]
[White "Van Foreest, Jorden"]
[Black "Grandelius, Nils"]
[Result "1-0"]
[WhiteFideId "1039784"]
[BlackFideId "1710400"]
[WhiteElo "2671"]
[BlackElo "2663"]
[WhiteClock "1:30:22"]
[BlackClock "0:31:57"]
[WhiteUrl "https://images.chesscomfiles.com/uploads/v1/master_player/0a158512-5d6d-11eb-b752-b5fd1d74df63.4770aa27.50x50o.7d0b0ee20d0b.jpeg"]
[WhiteCountry ""]
[WhiteTitle ""]
[BlackUrl "https://images.chesscomfiles.com/uploads/v1/master_player/23283f08-42da-11ea-9bab-d9c386462600.82f12f87.50x50o.12d6666d15fa.jpeg"]
[BlackCountry ""]
[BlackTitle ""]
[Link "https://www.chess.com/analysis/game/master/15790369?tab=analysis&move=19"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Qd3 $6 Nbd7 7. Be2 b5 8.
a4 Nc5 9. Qe3 b4 10. Nd5 Ncxe4 11. a5 $6 Nxd5 12. Qxe4 e6 $6 13. O-O $6 Bd7 $2 14.
Bd2 Be7 15. Bf3 O-O 16. Qd3 $1 Qb8 17. c4 bxc3 $1 18. bxc3 Ra7 $2 19. Rfb1 Qc8 $6 20.
c4 $1 Nf6 21. Nb5 $3 axb5 22. cxb5 Bxb5 23. Qxb5 Nd7 $2 24. Bb7 Qd8 25. a6 Bf6 26.
Ba5 Qe8 27. Bc7 $4 Bxa1 28. Rxa1 $6 d5 $2 29. Bd6 $1 Qd8 30. Rc1 g6 $2 31. h3 Re8 32.
Rc7 Nf6 33. Be5 Ne4 34. Qc6 Rf8 35. Bd4 Qb8 36. f3 Rxa6 37. Bxa6 Qb4 38. Be5
Qe1+ 39. Kh2 Nf2 40. Qc3 Qh1+ 41. Kg3 Qg1 42. Rc8 Nh1+ 43. Kh4 Qf2+ 44. g3 $1 g5+
45. Kxg5 f6+ $6 46. Kh6 $1 fxe5 47. Qxe5 1-0'''


if __name__ == '__main__':
    if sys.argv[1] == 'dd':
        from calc import Calculator

        if sys.platform == 'win32':
            STOCKFISHPATH = r'c:/gitproj/stockfish/stockfish-windows-x86-64-avx2.exe'
        else:
            STOCKFISHPATH = '/home/jovyan/stockfish/src/stockfish'
        mycalc = Calculator.from_engine_path(STOCKFISHPATH)
        mycalc.calc_entire_game(pgn)
        sys.exit(0)
    uci.sunfish=sunfish

    if sys.argv[1] == 'bb':
        fen='r2Rr1k1/pp3pp1/2p2bbp/4n3/4pBP1/2Q1P1NP/PPP2P2/1K3B1R b - - 0 17'
    elif sys.argv[1] == 'cc':
        fen=    'r2q1rk1/3nbppp/b1pp1n2/p3p3/1p1PP1P1/P5NP/1PP1NPB1/R1BQ1RK1 b - - 1 12' 
    if sys.argv[1]!='aa':
        from calc import Calculator
        if sys.platform=='win32':
            STOCKFISHPATH=r'c:/gitproj/stockfish/stockfish-windows-x86-64-avx2.exe'
        else:
            STOCKFISHPATH='/home/jovyan/stockfish/src/stockfish'

        def my_function(): 
            Calculator.UseCache=False
            mycalc=Calculator.from_engine_path(STOCKFISHPATH)
            tasks=[]
            for _ in range(2):  # Two rounds
                b=chess.Board(fen)
                start_time = timeit.default_timer()
                print(mycalc.ret_stats(b,0))
                end_time = timeit.default_timer()
                execution_time = end_time - start_time
                print(f"The function took {execution_time} seconds to execute")
            mycalc.printtimer()
            mycalc.end()

        my_function()




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
