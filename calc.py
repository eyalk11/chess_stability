from tabulate import tabulate
import threading
from functools import partial
from copy import deepcopy 
import os,re,sys
from multiprocessing import Pool
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
import time
from collections import defaultdict
from queue import Queue
import io

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
from simpleexceptioncontext import SimpleExceptionContext,simple_exception_handling 
uci.sunfish = sunfish
pool = None
import logging
def neverthrow(f,*args,default=None,**kwargs):
    try:
        return f(*args,**kwargs)
    except:
        return default


class MyFormatter(logging.Formatter):
    log_format = 'Run %(run_number)s | %(asctime)s | %(filename)s:%(lineno)d:%(function)s | %(levelname)s | %(message)s'
    run_number=None
    def __init__(self):
        super().__init__(MyFormatter.log_format)

    def format(self, record):
        record.run_number = self.run_number
        record.filename = os.path.basename(record.pathname)
        record.function = record.funcName
        record.lineno = record.lineno
        return super().format(record)

#add logging to file 
import logging
logfile='calc.log' 
logger = logging.getLogger() 
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler()) 
h=logging.FileHandler(logfile) 
h.setFormatter(MyFormatter())
logger.addHandler(h) 
last_run = 0
if logfile and MyFormatter.run_number is None:
    if os.path.exists(logfile):
        for z in open(logfile):
             last_run=max(last_run,neverthrow(lambda: int(re.search('Run (\d+) \|',z).group(1)),default=0))

    last_run+=1
    MyFormatter.run_number=last_run

logging.debug('Started')

try:
    from memoization import cached
except:    
    cached=None 


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


#@simple_exception_handling("dispboard")
def display_board(fen,iswhite,STOCKFISHPATH): #actually display fen
    mycalc=Calculator(STOCKFISHPATH)
    b=chess.Board(fen)
    if iswhite is None:
        iswhite= b.turn
    #display( widgets.HTML(str(b._repr_svg_())))

    return mycalc.ret_stats(b, iswhite)
    #return mycalc

class StabilityStats:
    def __init__ (self,extended_stat=False,half_move_number=2):
        self.same_stab_frq=0 
        self.diff_stab_frq=0 
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
        self.pv= []
        self.extended_stat=extended_stat
        self.half_move_number=half_move_number

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

    def format_move_with_numbering(self, moves, half_move_number):
        formatted_moves = []
        f=half_move_number%2
        move_number=half_move_number//2 +1

        for i, move in enumerate(moves):
            if i == 0 and f:
                formatted_moves.append(f"{move_number}. .. {move}")
                move_number += 1
            elif i % 2 == f:
                formatted_moves.append(f"{move_number}. {move}")
                move_number += 1
            else:
                formatted_moves.append(move)

        return ' '.join(formatted_moves)

    def format_stats(self):
        minimal_keys = ['score', 'stability all', 'stability same', 'stability diff', 'num of reasonable moves', 'moves by depth']
        dic_formats = {'score': '{:.2f}', 'stability all': '{:.2f}%', 'stability same': '{:.2f}%', 'stability diff': '{:.2f}%', 'same move frequency': '{:.2f}%', 'diff move frequency': '{:.2f}%', 'num of reasonable moves': '{}', 'max(score) of reasonable': '{}', 'min(score) of reasonable': '{}', 'mean': '{:.2f}', 'stdev': '{:.2f}', 'fraction method': '{}', 'moves by depth': '{}'}
        
        data = {
            'score': self.score / 100,
            'stability all': self.stability_all * 100,
            'stability same': self.stability_same * 100,
            'stability diff': self.stability_diff * 100,
            'same move frequency': self.same_stab_frq,
            'diff move frequency': self.diff_stab_frq,
            'num of reasonable moves': len(self.num_of_reasonable_moves),
            'max(score) of reasonable': max(self.max_score_of_reasonable),
            'min(score) of reasonable': min(self.min_score_of_reasonable),
            'mean': self.mean,
            'stdev': self.stdev,
            'fraction method': self.fraction_method,
            'moves by depth': self.moves_by_depth
        }
        if not self.extended_stat:
            data = {k: v for k, v in data.items() if k in minimal_keys}

        result_str = f"Score: {dic_formats['score'].format(data['score'])}\n"
        result_str += "Few available moves\n" if self.fraction_method else ""


        table2 = []
        table=[]
        for t,d in list(self.pv.items()):
            table2.append([f"{dic_formats['score'].format( d[0] /100 )}", f"{self.format_move_with_numbering( t.split(';') ,self.half_move_number )}",f"{'Y' if d[1] else ''}"])

        for i,(key, value) in enumerate(list(data.items())[1:]):
            table.append([key, dic_formats[key].format(value)] + (table2[i] if i<len(table2) else []))

        if len(table2)>len(table):
            for i in range(len(table),len(table2)):
                table.append(['','',table2[i][0],table2[i][1],table2[i][2]])

        result_str += tabulate(table, headers=['Stat', 'Value', 'SCORE' ,'PV','FOUND'], tablefmt='plain')
        return result_str

class Calculator(object):
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
    MINMOVES= 3
    ThreadPoolCount= 4
    MAXTIMEPV = 5
    PVDEPTH = 10 
    EXTENDED_STATS = False 

    def __new__(cls, *args, **kwargs):
        inst=object.__new__(cls)
        try:
            import yaml
            with open('config.yaml') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                if 'Calculator' in data:
                    for k,v in data['Calculator'].items():
                        setattr(cls,k,v)
                if 'Logging' in data:
                    logger.setLevel(getattr(logging, data['Logging']['level']))
        except:
            pass
        return inst

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
        self.positions = {}
        if self.UseCache: 
            self.get_score = cached(custom_key_maker=self.custom_key_maker_score)(self.get_score)
        if self.UseCache: 
            self.calc_stability = cached(custom_key_maker=self.custom_key_maker)(self.calc_stability)

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

    def custom_key_maker_score(self, b, white, weak=False, deprel=None,gid=None):
        return b.fen(), b.ply(), white, weak #three-fold things

    def get_pv(self,b,max_depth,curls):
        engine=self.enginedic['kkk'][0]
        res=engine.analyse(b,multipv=6,limit=chess.engine.Limit(depth=self.PVDEPTH,time=self.MAXTIMEPV))
        
        for i in res:
            sa= self.convert_to_san(b, i['pv'])
            sa = ';'.join(str(x) for x in sa) 
            curls.append((sa ,self.convert_score(i['score'],b.turn)))

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
        return self.convert_score(cc, white)

    def convert_score(self, cc, white):
        if type(cc.relative) is chess.engine.Mate or type(cc.relative) is chess.engine.MateGiven:
            if cc.relative < chess.engine.Cp(-500):
                return -5000
            else:
                return 5000

        if cc.relative.score() is None:
            print(cc)
            print(cc.score())
            #breakpoint()
        return cc.relative.score() * (1 if  white else (-1))

    def convert_to_san(self,b,moves):
        cur=b.copy() 
        for  z in moves:
            yield cur.san(z)
            cur.push(z)


    def get_mov_sunfish(self, mov, ply=0):
        move = mov.uci()
        i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
        if ply % 2 == 1:
            i, j = 119 - i, 119 - j
        return Move(i, j, prom)

    @timer.time_execution
    def calc_moves_score_worker(self, cur, white, curdep, maxdepth, oldeval, lastlevellen, tored, levelsmap, seq, reslist,legalmovesdic, score=0):
        def get_mov_val(mov):
            try:
                return pos.value(t := self.get_mov_sunfish(mov, not white))
            except:
                logger.error(f"Error in calc_moves_score_worker: {curfen} {mov}") 

        gid=str(randrange(0, 1000))
        oldeval = self.get_score(cur, white, curdep !=
                                 maxdepth, curdep / maxdepth,gid)
        curfen=cur.fen()

        pos = uci.from_fen(*curfen.split(" "))
        ls = list(cur.generate_legal_moves())
        legalmovesdic[curdep]+=len(ls)
        excepectnu = int(300 * 3 ** (curdep - 1))

        if tored or (curdep>=3 and len(ls)*lastlevellen>=excepectnu):
            ls = [(get_mov_val(m), m)  for m in ls]
            ls=list(filter(lambda x: x[0] is not None, ls) )
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
            moves.append((cur.copy(), ev, g,san))
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

        for (curb, ev, g, san) in moves:
            if curdep not in levelsmap:
                levelsmap[curdep] = []
            levelsmap[curdep].append((";".join(seq + [san]), ev, g))
            if curdep < maxdepth:
                yield (curb, not white, curdep + 1, maxdepth, ev, len(moves)
                               * lastlevellen, tored, levelsmap, seq + [san], reslist,legalmovesdic,score+g * (1 if white else -1)) 

            if (self.JustTop and curdep == maxdepth) or curdep >= maxdepth -1:
                reslist.append((ev,curdep,seq+ [san] ,san,score))

    def calc_moves_score(self, cur, white, oldeval, depth=4):
        reslist=[]
        levelsmap = dict()
         


        if self.pool is None:
            # creates a pool of cpu_count() processes
            self.pool = ThreadPool(Calculator.ThreadPoolCount)#cpu_count())
        legalmovesdic=defaultdict(lambda: 0) 

        ngen=self.calc_moves_score_worker(cur.copy(), white, 1, depth, oldeval,
                       1, False, levelsmap, [], reslist,legalmovesdic)

        def myfunc(x):
            if x is None:
                return []
            return list(self.calc_moves_score_worker(*x ))

        pv_moves=[] 
        fut=self.pool.apply_async(self.get_pv, (cur,depth,pv_moves))


        while True: 
            gen=ngen
            ngen=[]
            for k in self.pool.imap_unordered(myfunc,gen,1 ):
                ngen+=k
            if len(ngen)==0:
                break 

        fut.get()

        pv_moves_dic = dict( { y : (score,0) for y,score in pv_moves})

        for ev,curdep,seq,san,score in reslist: 
            s=';'.join(seq)
            for t in pv_moves_dic:
                if s.startswith(t[:len(s)]):
                    pv_moves_dic[t]=(pv_moves_dic[t][0],1)
                    break

        return reslist, levelsmap , legalmovesdic ,pv_moves_dic

    def print_stats(self, curb, iswhite, full=True):
        #if neverthrow(asyncio.get_event_loop) is None:
        asyncio.set_event_loop(asyncio.new_event_loop())

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self.async_print_stats(curb, iswhite, full))
        else:
            print(loop.run_until_complete(self.async_ret_stats(curb, iswhite, full)))


    async def async_ret_stats(self, curb, iswhite, full=True):
        with SimpleExceptionContext("async_ret_stats"):
            if not full:
                lev = self.get_score(curb, iswhite)
                mystr = f"score: {lev / 100}"
                return mystr
            else:
                return self.ret_stats(curb, iswhite)

    def ret_stats(self, curb, iswhite):
        result = self.calc_stability(curb, iswhite)
        return result.format_stats()


    async def async_print_stats(self, fen, iswhite, full=True):
        print(await self.async_ret_stats(fen, iswhite, full))

    @staticmethod
    def get_game(pgn):
        p=chess.pgn.read_game(io.StringIO(pgn))
        gam=p.game()
        return gam

    @simple_exception_handling("calc_entire")
    def calc_entire_game(self,pgn):
        def with_p(k,san,*args):
            #tt=await self.async_ret_stats(*args)
            tt=display_board(*args)
            return k,san, tt
        gam = Calculator.get_game(pgn)

        g=deepcopy(gam)

        ls=[] 
        k=0
        with ProcessPoolExecutor(max_workers=7) as executor:
            while True:
                k+=1
                prev=g.board()
                g=g.next()
                if not g:
                    break
                b=g.board()
                san= prev.san(g.move )

                if k>self.MINMOVES:
                    f = executor.submit(display_board,b.fen(),g.turn,self.path)#(partial(with_p,k,san ) , b.fen(), g.turn)
                    ls.append(f)

        for t in ls:
            print(t.result())
        #results =  asyncio.gather(*(tuple(ls)))
        #r=results.result()
        #r.sort()
        #return r




    def custom_key_maker(self,curb, iswhite):
        return curb.fen(), curb.ply(), iswhite #three-fold things

    # @optional_decorator(cached, UseCache,custom_key_maker=custom_key_maker)
    def calc_stability(self, cur_board, iswhite):
        '''
        calcs stability by first getting score of all reasonable moves, then apply calculation(see readme).
        returns initial score, vector of diff vs initial score, stability factor, if fraction method
        '''
        # logger.info(f"calc_stability {cur_board.fen()} {iswhite}")
        r=StabilityStats(self.EXTENDED_STATS,cur_board.ply() )
        init = self.get_score(cur_board, iswhite)
        init = self.get_score(cur_board, iswhite)
        init = self.get_score(cur_board, iswhite)
        arr, lev, legalmovesdic,pv_moves_dic  = self.calc_moves_score(cur_board, iswhite, init )
        r.pv= pv_moves_dic 

        nlev=lev.copy() 
        if type(nlev[1] ) is not int:
            nlev = { k: len(v) for k,v in nlev.items()} 

        maxlev=max(nlev.keys())
        #better movfraq 
        movfraq={x: nlev[x]/legalmovesdic[x] for x in legalmovesdic if x in nlev} 
        r.same_stab_frq=movfraq[maxlev  ] * 100 if len(movfraq)%2==0 else movfraq[maxlev-1]
        r.diff_stab_frq=movfraq[maxlev] * 100 if len(movfraq)%2==1 else movfraq[maxlev-1]


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
        logging.debug(f"stabsame:{stabsame},stabdiff:{stabdiff},stab:{stab},pop_stdev:{pop_stdev},tactical:{tactical},nlev:{nlev}")
        r.assign(init, stab, stabsame, stabdiff, arr, data, data, data.mean(), pop_stdev, tactical, nlev if len(str(dict(lev))) > 300 else lev)
        
        return r
