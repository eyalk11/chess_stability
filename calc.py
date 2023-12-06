from collections import defaultdict
import asyncio
#b = Button(description='HideShow')
import chess.engine
import numpy as np
from chess import Board
import chess.engine
STOCKFISHDEPTH=10
SIGMA= 5
FRACFACTOR=2.5
SCORECUTOFF=200
class Calculator:
    def __init__(self,engine):
        self.engine = engine

    def get_score(self,b, white):
        cc = self.engine.analyse(b, chess.engine.Limit(depth=STOCKFISHDEPTH))['score']

        if type(cc.relative) is chess.engine.Mate or type(cc.relative) is chess.engine.MateGiven:
            if cc.relative < chess.engine.Cp(-500):
                return -5000
            else:
                return 5000

        if cc.relative.score() is None:
            print(cc)
            print(cc.score())
            breakpoint()
        return cc.relative.score() * (1 if white else (-1))



    async def calc_moves_score_int(self,cur, white, curdep, maxdepth, oldeval, l):
        from aiostream import stream
        fen = cur.fen()
        from aiostream import stream, pipe

        async with stream.iterate(cur.generate_legal_moves()).stream() as it:
            async for z in it:  # stream.list(cur.generate_legal_moves()):
                cur = Board(fen=fen)
                cur.push(z)
                ev = self.get_score(cur, not white)
                g = ev - oldeval
                if white and g < (-1) * SCORECUTOFF:  # too bad
                    cur.pop()
                    continue
                elif g > SCORECUTOFF:
                    cur.pop()
                    continue

                l[curdep] += 1

                if curdep == maxdepth:
                    yield ev
                else:
                    async for k in self.calc_moves_score_int(cur, not white, curdep + 1, maxdepth, ev, l):
                        yield k

                cur.pop()


    def print_stats(self,curb, iswhite):
        try:

            init, origg, stab, b, lev = self.calc_stability(curb, iswhite)
            ls = ['score', 'stability factor', 'num of reasonable moves', 'max(score) of reasonable',
                  'min(score)  of reasonable', 'faraction method', 'moves by depth']
            tup = ('%.2f' % (init / 100), ('%.2f' % (stab * 100)) + '%', len(origg), max(origg), min(origg), b, dict(lev))
            for a, b in zip(ls, tup):
                print(a + ': ' + str(b))
        except Exception as e:
            import traceback
            print(traceback.format_exc())


    def calc_moves_score(self,cur, white, oldeval, depth=2):
        from aiostream import stream

        async def tmp():
            xx = await stream.list(self.calc_moves_score_int(cur, white, 1, depth, oldeval, lev))
            return xx

        if asyncio.get_event_loop() is None:
            asyncio.set_event_loop(asyncio.new_event_loop())
        lev = defaultdict(int)
        ll = asyncio.run(tmp())
        return ll, lev


    def calc_stability(self,cur_board, iswhite):
        '''
        calcs stability by first getting score of all reasonable moves, then apply calculation(see readme).
        returns initial score, vector of diff vs initial score, stability factor, if fraction method
        '''
        init = self.get_score(cur_board, iswhite)
        arr, lev = self.calc_moves_score(cur_board, iswhite, init)
        g = np.array(arr, dtype="float64")
        g -= init
        g /= 100

        orig_vec = np.copy(g)
        b = False
        if abs(init) > 200:
            b = True
            g = g / (init / 100) * FRACFACTOR  # TODO:to make continous...
        else:
            b = False
        # We only care about worsing moves, every good move can't contribute more than 1
        if iswhite:
            g[g > 0] = 0
        else:
            g[g < 0] = 0

        g = (-1) * np.square(g) * SIGMA

        g = np.exp(g)
        stab = np.sum(g) / len(g)
        if stab > 0.99:
            print(g)
            print(list(orig_vec))
        return init, orig_vec, stab, b, lev
