from collections import defaultdict
import asyncio
import chess.engine
import numpy as np
from chess import Board
from aiostream import stream
from memoization import cached

from random import randrange 


from sunfish import sunfish
from sunfish.sunfish import render
from sunfish.tools import uci
from sunfish.sunfish import parse,Move
from multiprocessing import cpu_count  
uci.sunfish= sunfish 


class Calculator:
    STOCKFISHDEPTH = 10
    SIGMA = 5
    FRACFACTOR = 2.5
    SCORECUTOFF = 100
    STOCKFISHDEPTHWEAK = 3
    ELOWEAK = 1620
    SCORETHR = 40
    DODEPTH = 3
    WEAKTIME= 0.004
    STOCKFISHSTRONG=0.04  
    @classmethod
    def from_engine_path(cls, path):
        return Calculator(path)

    @classmethod
    def eng_from_engine_path(cls, path):
        engine = chess.engine.SimpleEngine.popen_uci(path)
        weak = chess.engine.SimpleEngine.popen_uci(path)
        # weak.configure({"UCI_LimitStrength": True, "UCI_Elo": cls.ELOWEAK})
        return (engine, weak)

    def __init__(self, path):
        self.path = path
        self.enginedic = defaultdict(
                lambda: Calculator.eng_from_engine_path(self.path))
        self.gameidf = str(randrange(0, 1000))
        self.cpucores = cpu_count()
        self.posdic= {} 

    def get_score(self, b, white,weak=False,deprel=None):
        if deprel is None:
            f=1
        else:
            f= 2** (-10* deprel)
        process_name = randrange(0, self.cpucores)

        limit = (
            chess.engine.Limit(depth=self.STOCKFISHDEPTH, time=self.STOCKFISHSTRONG * f * 1000)
            if not weak
            else chess.engine.Limit(depth=self.STOCKFISHDEPTHWEAK, time=self.WEAKTIME * f * 1000  ,nodes=6800 * f * 1000)
        )
        engine = self.enginedic[process_name][0] if not weak else self.enginedic[process_name][1]
        if not weak:
            cc = engine.analyse(b, limit, game=self.gameidf)["score"]
        else:
            cc = engine.analyse(b, limit,game = str(randrange(0, 1000)) )["score"]

        #cc = engine.analyse(b, chess.engine.Limit(depth=self.STOCKFISHDEPTH if not weak else self.STOCKFISHDEPTHWEAK ),game=self.gameidf)['score']

        if type(cc.relative) is chess.engine.Mate or type(cc.relative) is chess.engine.MateGiven:
            if cc.relative < chess.engine.Cp(-500):
                return -5000
            else:
                return 5000

        if cc.relative.score() is None:
            print(cc)
            print(cc.score())
            breakpoint()
        return cc.relative.score() * (1 if not white else (-1))

    def get_mov_sunfish(self,mov,ply=0):
        move=mov.uci()

        i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
        if ply % 2 == 1:
            i, j = 119 - i, 119 - j
        return Move(i, j, prom)

    async def calc_moves_score_int(self, fen, white, curdep, maxdepth, oldeval, original_score, levelsmap,seq=[],lastlevellen=0,tored=False):
        # fen = cur.fen()
        cur = Board(fen=fen)
        oldeval = self.get_score(cur, white, curdep != maxdepth,curdep/maxdepth)
        oldeval = self.get_score(cur, white, curdep != maxdepth, curdep / maxdepth)

        pos = uci.from_fen(*fen.split(" "))
        ls = list(cur.generate_legal_moves())
        if tored: 
            ls= [(pos.value(t:=self.get_mov_sunfish(m,not white)), m) for m in ls]
            ls = sorted(ls, reverse=True,key=lambda x:x[0]) #
            ls = list(filter(lambda x: x[0] < self.SCORETHR, ls))
            ls= ls[:5]
        else:
            ls= [(1, m) for m in ls] 


        async def generate_moves():
            async with stream.iterate(ls).stream() as it:
                async for _,z in it:
                    cur = Board(fen=fen)
                    #cur.turn=white
                    san = cur.san(z)
                    cur.push(z)
                    ev = self.get_score(cur, not white, curdep != maxdepth,curdep/maxdepth)
                    g = ev - oldeval
                    if g < (-1) * self.SCORECUTOFF:  # too bad
                        cur.pop()
                        continue
                    elif g > self.SCORECUTOFF:
                        cur.pop()
                        continue
                    yield cur.fen(), ev, g, san 
                    cur.pop()

        gen= [(curfen, ev, g, san) async for (curfen, ev, g, san) in generate_moves()]

        if curdep > 2:
            excepectnu= int(10 * 5**((curdep-1)))  
            curlev= lastlevellen * len(gen)
            if curlev  > excepectnu:
                print('reduce', curdep, curlev, excepectnu)
                tored=True
                #reduce for best expceptnu best ev 
                gen= sorted(gen, key=lambda x: x[1], reverse=True) 
                gen= gen[:excepectnu] 
                # ls = list(filter(lambda x: x[0] < self.SCORETHR, ls))
                # nu= int(40 * 5**((-1)*(curdep-1)))
                if curdep<maxdepth: 
                    maxdepth -=1

        for (curfen, ev, g, san) in gen:

                levelsmap[curdep].add((";".join(seq+ [san]), ev,g))
                #print((";".join(seq+ [san]), ev,g))

                if curdep == maxdepth:
                    yield ev
                else:
                    async for k in self.calc_moves_score_int(curfen, not white, curdep + 1, maxdepth, ev, original_score, levelsmap,seq + [san],len(gen)*lastlevellen ,tored=False):
                        yield k


    async def async_print_stats(self, curb, iswhite, full=True):
        if not full:
            lev= self.get_score(curb, iswhite)
            print( f"score: {lev/100}")
            return

        try:
            init, origg, stab, b, lev = await self.calc_stability(curb, iswhite )
            mbydepth = dict(lev)
            if len(str(dict(lev)))> 2000:
                mbydepth = {k: len(v) for k, v in lev.items()} 
            ls = ['score', 'stability factor', 'num of reasonable moves', 'max(score) of reasonable',
                  'min(score)  of reasonable', 'faraction method', 'moves by depth']
            tup = ('%.2f' % (init / 100), ('%.2f' % (stab * 100)) + '%', len(origg), max(origg), min(origg), b, mbydepth)
            for a, b in zip(ls, tup):
                print(a + ': ' + str(b))
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    async def async_calc_moves_score(self, cur, white, oldeval,  depth=4):
        async def tmp():
            xx = await stream.list(self.calc_moves_score_int(cur.fen(), white, 1, depth, oldeval, oldeval, lev))
            return xx

        lev = defaultdict(set)
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
        init = self.get_score(cur_board, iswhite)
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
        #geometric mean of
        stab = g.prod() ** (1 / len(g))
        #np.mean()
        #stab = np.sum(g) / len(g)
        if stab > 0.99:
            print(g)
            print(list(orig_vec))
        return init, orig_vec, stab, b, lev
