from collections import defaultdict
import asyncio
import chess.engine
import numpy as np
from chess import Board
from aiostream import stream
from memoization import cached

from random import randrange 
class Calculator:
    STOCKFISHDEPTH = 10
    SIGMA = 5
    FRACFACTOR = 2.5
    SCORECUTOFF = 200
    STOCKFISHDEPTHWEAK = 3

    def __init__(self,engine):
        self.engine = engine
        self.gameidf= str(randrange(0,1000) )

    @cached
    def get_score(self, b, white,weak=False):
        cc = self.engine.analyse(b, chess.engine.Limit(depth=self.STOCKFISHDEPTH if not weak else self.STOCKFISHDEPTHWEAK ),game=self.gameidf)['score']

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

    async def calc_moves_score_int(self, cur, white, curdep, maxdepth, oldeval, original_score, levelsmap):
        fen = cur.fen()

        async with stream.iterate(cur.generate_legal_moves()).stream() as it:
            async for z in it:
                cur = Board(fen=fen)
                cur.push(z)
                ev = self.get_score(cur, not white, curdep != maxdepth)
                g = ev - oldeval
                if white and g < (-1) * self.SCORECUTOFF:  # too bad
                    cur.pop()
                    continue
                elif g > self.SCORECUTOFF:
                    cur.pop()
                    continue

                levelsmap[curdep] += 1

                if curdep == maxdepth:
                    yield ev
                else:
                    async for k in self.calc_moves_score_int(cur, not white, curdep + 1, maxdepth, ev, original_score, levelsmap):
                        yield k

                cur.pop()

    async def async_print_stats(self, curb, iswhite, full=True):
        if not full:
            lev= self.get_score(curb, iswhite)
            print( f"score: {lev/100}")
            return

        try:
            init, origg, stab, b, lev = await self.calc_stability(curb, iswhite )
            ls = ['score', 'stability factor', 'num of reasonable moves', 'max(score) of reasonable',
                  'min(score)  of reasonable', 'faraction method', 'moves by depth']
            tup = ('%.2f' % (init / 100), ('%.2f' % (stab * 100)) + '%', len(origg), max(origg), min(origg), b, dict(lev))
            for a, b in zip(ls, tup):
                print(a + ': ' + str(b))
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    async def async_calc_moves_score(self, cur, white, oldeval,  depth=2):
        async def tmp():
            xx = await stream.list(self.calc_moves_score_int(cur, white, 1, depth, oldeval, oldeval, lev))
            return xx

        lev = defaultdict(int)
        ll = await tmp()
        return ll, lev

    def print_stats(self, curb, iswhite,full=True):
        if asyncio.get_event_loop() is None:
            asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.run(self.async_print_stats( curb, iswhite, full))

    async def calc_stability(self,cur_board, iswhite):
        '''
        calcs stability by first getting score of all reasonable moves, then apply calculation(see readme).
        returns initial score, vector of diff vs initial score, stability factor, if fraction method
        '''
        init = self.get_score(cur_board, iswhite)
        arr, lev = await self.async_calc_moves_score(cur_board, iswhite, init)
        g = np.array(arr, dtype="float64")
        g -= init
        g /= 100

        orig_vec = np.copy(g)
        b = False
        if abs(init) > 200:
            b = True
            g = g / (init / 100) * self.FRACFACTOR  # TODO:to make continous...
        else:
            b = False
        # We only care about moves that make things worse, every good move can't contribute more than 1
        if iswhite:
            g[g > 0] = 0
        else:
            g[g < 0] = 0

        g = (-1) * np.square(g) * self.SIGMA

        g = np.exp(g)
        stab = np.sum(g) / len(g)
        if stab > 0.99:
            print(g)
            print(list(orig_vec))
        return init, orig_vec, stab, b, lev
