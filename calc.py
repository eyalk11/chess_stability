import threading
from multiprocessing import Pool
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import time
from collections import defaultdict
from queue import Queue
import chess.engine
import numpy as np
from chess import Board
from multiprocessing import  Pool, Manager, cpu_count
from multiprocessing.pool import ThreadPool
import asyncio
from random import randrange
from sunfish import sunfish
from sunfish.sunfish import render
from sunfish.tools import uci
from sunfish.sunfish import parse,Move
from multiprocessing import cpu_count  
from sunfish.sunfish import parse, Move
import multiprocessing as mp
uci.sunfish = sunfish
pool = None
try:
    from memozation import cached
except:    
    cached=None 

from functools import wraps

def optional_decorator(decorator,condition,*args,**kwargs):
    def apply_decorator(func):
        if condition and decorator is not None:  # Replace CONDITION with your condition
            return decorator(func)
        return func
    return apply_decorator

try:
    from execution_timer import ExecutionTimer
except:
    class ExecutionTimer:
        @staticmethod
        def time_execution(f):
            """Decorate by doing nothing."""
            def decorated_function(*args, **kwargs):
                return f(*args, **kwargs)
            return decorated_function
        def average_measured_time(self):
            return {}

timer = ExecutionTimer()

class ReturnValues:
    def __init__ (self): 
        self.score = 0
        self.stability_all = 0
        self.stability_same = 0
        self.stability_diff = 0
        self.num_of_reasonable_moves = []
        self.max_score_of_reasonable = []
        self.min_score_of_reasonable = []
        self.mean = 0
        self.stdev = 0
        self.fraction_method = False
        self.moves_by_depth = [] 

    def assign(self, score, stability_all, stability_same, stability_diff, num_of_reasonable_moves, max_score_of_reasonable,
                 min_score_of_reasonable, mean, stdev, fraction_method, moves_by_depth):
        self.score = score
        self.stability_all = stability_all
        self.stability_same = stability_same
        self.stability_diff = stability_diff
        self.num_of_reasonable_moves = num_of_reasonable_moves
        self.max_score_of_reasonable = max_score_of_reasonable
        self.min_score_of_reasonable = min_score_of_reasonable
        self.mean = mean
        self.stdev = stdev
        self.fraction_method = fraction_method
        self.moves_by_depth = moves_by_depth

    def format_stats(self):
        ls = ['score', 'stability all', 'stability same','stability diff',  'num of reasonable moves', 'max(score) of reasonable',
              'min(score) of reasonable','mean' , 'stdev', 'fraction method', 'moves by depth']
        tup = (f"{self.score / 100:.2f}",
               f"{self.stability_all * 100:.2f}%",
               f"{self.stability_same * 100:.2f}%",
               f"{self.stability_diff * 100:.2f}%",
               len(self.num_of_reasonable_moves),
               max(self.max_score_of_reasonable),
               min(self.min_score_of_reasonable),
               f"{self.mean:.2f}",
               f"{self.stdev:.2f}",
               self.fraction_method,
               self.moves_by_depth)
        mystr = "few available moves\n" if self.fraction_method else ""
        for a, b in zip(ls, tup):
            mystr += f"{a}: {b}\n"
        return mystr

class Calculator:
    STOCKFISHDEPTH = 10
    SIGMA = 5
    FRACFACTOR = 2.5
    SCORECUTOFF = 80
    STOCKFISHDEPTHWEAK = 3
    ELOWEAK = 1620
    SCORETHR = 40
    DODEPTH = 3
    WEAKTIME= 0.02
    STOCKFISHSTRONG=0.4
    UseWeakElo=False
    UseCache=True
    JustTop=False

    @classmethod
    def from_engine_path(cls, path):
        return Calculator(path)

    @classmethod
    def eng_from_engine_path(cls, path):
        engine = chess.engine.SimpleEngine.popen_uci(path)
        weak = chess.engine.SimpleEngine.popen_uci(path)
        if cls.UseWeakElo:
            weak.configure({"UCI_LimitStrength": True, "UCI_Elo": cls.ELOWEAK})
        return (engine, weak)

    def __init__(self, path):

        self.path = path
        self.gameidf = str(randrange(0, 1000))
        self.posdic = {}
        self.enginedic = defaultdict(
            lambda: Calculator.eng_from_engine_path(self.path))
        self.pool = None

    def ret_timer(self):
        global timer
        return (timer.average_measured_time())

    def printtimer(self):
        global timer
        print(timer.average_measured_time())


    def end(self):
        for k, v in self.enginedic.items():
            v[0].close()
            v[1].close()

    @timer.time_execution
    def get_score(self, b, white, weak=False, deprel=None,gid=None):
        try:
            process_name = threading.current_thread().ident

        except:
            process_name = 'nnn'
            # engine= self.weakengine if weak else self.engine
        engine = self.enginedic[process_name][0] if not weak else self.enginedic[process_name][1]

        if deprel is None:
            f = 1
        else:
            f = 2 ** (-10 * deprel)

        limit = (
            chess.engine.Limit(depth=self.STOCKFISHDEPTH,
                               time=self.STOCKFISHSTRONG * f * 1000)
            if not weak
            else chess.engine.Limit(depth=self.STOCKFISHDEPTHWEAK, time=self.WEAKTIME * f * 1000)
        )
        # engine = self.weakengine if weak else self.engine
        if not weak:
            cc = engine.analyse(b, limit, game=self.gameidf)["score"]
        else:
            cc = engine.analyse(b, limit, game=gid)["score"]

        if type(cc.relative) is chess.engine.Mate or type(cc.relative) is chess.engine.MateGiven:
            if cc.relative < chess.engine.Cp(-500):
                return 5000
            else:
                return -5000

        if cc.relative.score() is None:
            print(cc)
            print(cc.score())
            breakpoint()
        return cc.relative.score() * (1 if not white else (-1))

    def get_mov_sunfish(self, mov, ply=0):
        move = mov.uci()
        i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
        if ply % 2 == 1:
            i, j = 119 - i, 119 - j
        return Move(i, j, prom)

    @timer.time_execution
    def calc_moves_score_worker(self, curfen, white, curdep, maxdepth, oldeval, lastlevellen, tored, levelsmap, seq, reslist,legalmovesdic, score=0):
        gid=str(randrange(0, 1000))
        cur = Board(fen=curfen)
        oldeval = self.get_score(cur, white, curdep !=
                                 maxdepth, curdep / maxdepth,gid)

        pos = uci.from_fen(*curfen.split(" "))
        ls = list(cur.generate_legal_moves())
        legalmovesdic[curdep]+=len(ls)
        excepectnu = int(300 * 3 ** (curdep - 1))
        if tored or (curdep>=3 and len(ls)*lastlevellen>=excepectnu):
            ls = [(pos.value(t := self.get_mov_sunfish(m, not white)), m)
                  for m in ls]
            ls.sort( reverse=True, key=lambda x: x[0])
            ls = list(filter(lambda x: x[0] < self.SCORETHR, ls))
            ls = ls[:5]
        else:
            ls = [(1, m) for m in ls]

        moves = []
        for _, z in ls:
            san =  cur.san(z)
            cur.push(z)
            ev = self.get_score(cur, not white, curdep !=
                                maxdepth, curdep / maxdepth,gid)
            g = ev - oldeval
            if g < (-1) * self.SCORECUTOFF or g > self.SCORECUTOFF:
                cur.pop()
                continue
            moves.append((cur.fen(), ev, g,san))
            cur.pop()

        # apply expectednu filter
        if curdep > 2:
            excepectnu = int(10 * 5 ** (curdep - 1))/2
            curlev = lastlevellen * len(moves)
            if curlev > excepectnu:
                tored = True
                moves.sort(key=lambda x: x[2], reverse=True)
                e=int(excepectnu/4)
                moves= moves[:e]
                if curdep < maxdepth:
                    maxdepth -= 1

        for (curfen, ev, g, san) in moves:
            if curdep not in levelsmap:
                levelsmap[curdep] = []
            levelsmap[curdep].append((";".join(seq + [san]), ev, g))
            if curdep < maxdepth:
                yield (curfen, not white, curdep + 1, maxdepth, ev, len(moves)
                               * lastlevellen, tored, levelsmap, seq + [san], reslist,legalmovesdic,score+g * (1 if white else -1)) 

            if (self.JustTop and curdep == maxdepth) or curdep >= maxdepth -1:
                reslist.append((ev,curdep,seq ,san,score))

    def calc_moves_score(self, cur, white, oldeval, depth=4):
        reslist=[]
        levelsmap = dict()

        if self.pool is None:
            # creates a pool of cpu_count() processes
            self.pool = ThreadPool(4)#cpu_count())
        legalmovesdic=defaultdict(lambda: 0) 

        ngen=self.calc_moves_score_worker(cur.fen(), white, 1, depth, oldeval,
                       1, False, levelsmap, [], reslist,legalmovesdic)

        def myfunc(x):
            if x is None:
                return []
            return list(self.calc_moves_score_worker(*x ))


        while True: 
            gen=ngen
            ngen=[]
            for k in self.pool.imap_unordered(myfunc,gen,1 ):
                ngen+=k
            if len(ngen)==0:
                break 


        return reslist, levelsmap , legalmovesdic

    def print_stats(self, curb, iswhite, full=True):
        if asyncio.get_event_loop() is None:
            asyncio.set_event_loop(asyncio.new_event_loop())

        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = asyncio.ensure_future(self.async_print_stats(curb, iswhite, full))
        else:
            loop.run_until_complete(self.async_print_stats(curb, iswhite, full))

    def custom_key_maker(self, curb, iswhite, full=True):
        return curb.fen(), iswhite, full 

    @optional_decorator(cached, UseCache,custom_key_maker=custom_key_maker) 
    async def async_ret_stats(self, curb, iswhite, full=True):
        if not full:
            lev = self.get_score(curb, iswhite)
            mystr = f"score: {lev / 100}"
            return mystr

        try:
            result = await self.calc_stability(curb, iswhite)
            return result.format_stats()

        except Exception:
            import traceback
            print(traceback.format_exc())

    async def async_print_stats(self, fen, iswhite, full=True):
        print(await self.async_ret_stats(fen, iswhite, full))

    async def calc_stability(self, cur_board, iswhite):
        '''
        calcs stability by first getting score of all reasonable moves, then apply calculation(see readme).
        returns initial score, vector of diff vs initial score, stability factor, if fraction method
        '''
        r=ReturnValues() 
        init = self.get_score(cur_board, iswhite)
        init = self.get_score(cur_board, iswhite)
        init = self.get_score(cur_board, iswhite)
        arr, lev, legalmovesdic = self.calc_moves_score(cur_board, iswhite, init)
        nlev=lev.copy() 
        if type(nlev[1] ) is not int:
            nlev = { k: len(v) for k,v in nlev.items()} 
        #better movfraq 
        movfraq={x: nlev[x]/legalmovesdic[x] for x in legalmovesdic} 


        movfraq=(nlev[1]/legalmovesdic[1] + nlev[2]/legalmovesdic[2])/2
        tactical= (movfraq< 1/ 30) or ((nlev[1]+nlev[2]< 20))

        def calc_with_arr(arr,iswhite):
            g = np.array(arr, dtype="float64")
            g -= init
            g /= 100

            b = False
            if abs(init) > 200:
                b = True
                # TODO:to make continous...
                g = g / (init / 100) * self.FRACFACTOR 
            else:
                b = False
            # We only care about moves that make things worse, every good move can't contribute more than 1
            if iswhite:
                g[g > 0] = 0
            else:
                g[g < 0] = 0

            g = (-1) * np.square(g) * self.SIGMA
            g = np.exp(g)
            # geometric mean doesn't work well since sensitive to bad moves
            # stab = g.prod() ** (1 / len(g)) too sensitive
            # np.mean()
            stab = np.sum(g) / len(g)
            return stab,b 

        data = np.array([x[0] for x in arr], dtype="float64")
        pop_stdev = np.sqrt(np.mean((data - init)**2))
        same = [x[0] for x in arr if x[1] % 2 == 0 ]
        diff = [x[0] for x in arr if x[1] % 2 == 1 ]
        stabsame, b = calc_with_arr(same,iswhite)
        stabdiff, b2 = calc_with_arr(diff,not iswhite)
        b=b2 or b
        #if one of them is nan take the other one 
        if np.isnan(stabsame):
            stab=stabdiff
        elif np.isnan(stabdiff):
            stab=stabsame
        else:
            stab = (stabsame* len(same) + stabdiff * len(diff)) / len(arr)

        r.assign(init, stab, stabsame, stabdiff, arr, data, data, data.mean(), pop_stdev, tactical, lev)
        
        return r
