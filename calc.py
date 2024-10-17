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
from stability_stats import StabilityStats
from simpleexceptioncontext import SimpleExceptionContext,simple_exception_handling 
uci.sunfish = sunfish
pool = None
import logging
def never_throw(function, *args, default=None, **kwargs):
    try:
        return function(*args, **kwargs)
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
def start_logging():
    log_file='calc.log' 
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler()) 
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(MyFormatter())
    logger.addHandler(file_handler)
    last_run_number = 0
    if log_file and MyFormatter.run_number is None:
        if os.path.exists(log_file):
            for line in open(log_file):
                 last_run_number = max(last_run_number, never_throw(lambda: int(re.search('Run (\d+) \|', line).group(1)), default=0))

        last_run_number += 1
        MyFormatter.run_number = last_run_number

start_logging()
logging.info('Started')

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
def display_board(board, is_white, STOCKFISHPATH):
    mycalc = Calculator(STOCKFISHPATH)
    board_copy = board.copy()
    if is_white is None:
        is_white = board_copy.turn
    #display( widgets.HTML(str(b._repr_svg_())))

    return mycalc.ret_stats(board_copy, is_white)
    #return mycalc

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
    INCLUDE_PV = True  # New parameter added
    IGNORE_CONFIG=False 

    _config_loaded = False

    @classmethod
    def load_config(cls):
        if not cls._config_loaded and not cls.IGNORE_CONFIG:
            try:
                import yaml
                with open('config.yaml') as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                    if 'Calculator' in data:
                        for k, v in data['Calculator'].items():
                            setattr(cls, k, v)
                    if 'Logging' in data:
                        logger.setLevel(getattr(logging, data['Logging']['level']))
                cls._config_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

    def __new__(cls, *args, **kwargs):
        cls.load_config()
        return super().__new__(cls)

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
        self.game_id = str(randrange(0, 1000))
        self.position_dict = {}
        self.engine_dict = defaultdict(lambda: Calculator.eng_from_engine_path(self.path))
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
        for k, v in self.engine_dict.items():
            v[0].close()
            v[1].close()

    def custom_key_maker_score(self, b, white, weak=False, deprel=None,gid=None):
        return b.fen(), b.ply(), white, weak #three-fold things

    def get_pv(self,b,max_depth,curls):
        engine=self.engine_dict['kkk'][0]
        res=engine.analyse(b,multipv=6,limit=chess.engine.Limit(depth=self.PVDEPTH,time=self.MAXTIMEPV))
        
        for i in res:
            sa= self.convert_to_san(b, i['pv'])
            sa = ';'.join(str(x) for x in sa) 
            curls.append((sa ,self.convert_score(i['score'],b.turn)))

    @timer.time_execution
     
    def get_score(self, board, is_white, weak=False, depth_relative=None, game_id=None):
        try:
            process_name = threading.current_thread().ident
        except:
            process_name = 'nnn'

        engine = self.engine_dict[process_name][0] if not weak else self.engine_dict[process_name][1]

        if depth_relative is None:
            factor = 1
        else:
            factor = 2 ** (-10 * depth_relative)

        limit = (
            chess.engine.Limit(depth=self.STOCKFISHDEPTH, time=self.STOCKFISHSTRONG * factor * 1000)
            if not weak
            else chess.engine.Limit(depth=self.STOCKFISHDEPTHWEAK, time=self.WEAKTIME * factor * 1000)
        )

        if not weak:
            score = engine.analyse(board, limit, game=self.game_id)["score"]
        else:
            score = engine.analyse(board, limit, game=game_id)["score"]
        return self.convert_score(score, is_white)

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
    def calc_moves_score_worker(self, current_board, is_white, current_depth, max_depth, previous_eval, 
                                last_level_length, to_reduce, levels_map, move_sequence, result_list, 
                                legal_moves_count, cumulative_score=0):
        def get_move_value(move):
            try:
                return position.value(t := self.get_mov_sunfish(move, not is_white))
            except:
                logger.error(f"Error in calc_moves_score_worker: {current_fen} {move}") 

        game_id = str(randrange(0, 1000))
        current_eval = self.get_score(current_board, is_white, current_depth != max_depth, current_depth / max_depth, game_id)
        current_fen = current_board.fen()

        position = uci.from_fen(*current_fen.split(" "))
        legal_moves = list(current_board.generate_legal_moves())
        legal_moves_count[current_depth] += len(legal_moves)
        expected_num = int(300 * 3 ** (current_depth - 1))

        if to_reduce or (current_depth >= 3 and len(legal_moves) * last_level_length >= expected_num):
            move_values = [(get_move_value(move), move) for move in legal_moves]
            move_values = list(filter(lambda x: x[0] is not None, move_values))
            move_values.sort(reverse=True, key=lambda x: x[0])
            move_values = list(filter(lambda x: x[0] < self.SCORETHR, move_values))
            move_values = move_values[:5]
        else:
            move_values = [(1, move) for move in legal_moves]

        analyzed_moves = []
        for _, move in move_values:
            san = current_board.san(move)
            current_board.push(move)
            new_eval = self.get_score(current_board, not is_white, current_depth != max_depth, current_depth / max_depth, game_id)
            eval_diff = new_eval - current_eval
            if eval_diff < (-1) * self.SCORECUTOFF or eval_diff > self.SCORECUTOFF:
                current_board.pop()
                continue
            analyzed_moves.append((current_board.copy(), new_eval, eval_diff, san))
            current_board.pop()

        # Apply expected number filter
        if current_depth > 2:
            expected_num = int(10 * 5 ** (current_depth - 1)) / 2
            current_level = last_level_length * len(analyzed_moves)
            if current_level > expected_num:
                to_reduce = True
                analyzed_moves.sort(key=lambda x: x[2], reverse=True)
                limit = int(expected_num / 4)
                analyzed_moves = analyzed_moves[:limit]
                if current_depth < max_depth:
                    max_depth -= 1

        for (board, eval, eval_diff, san) in analyzed_moves:
            if current_depth not in levels_map:
                levels_map[current_depth] = []
            levels_map[current_depth].append((";".join(move_sequence + [san]), eval, eval_diff))
            if current_depth < max_depth:
                yield (board, not is_white, current_depth + 1, max_depth, eval, len(analyzed_moves) * last_level_length, 
                       to_reduce, levels_map, move_sequence + [san], result_list, legal_moves_count, 
                       cumulative_score + eval_diff * (1 if is_white else -1))

            if (self.JustTop and current_depth == max_depth) or current_depth >= max_depth - 1:
                result_list.append((eval, current_depth, move_sequence + [san], san, cumulative_score))

    def calc_moves_score(self, current_board, is_white, old_eval, depth=4):
        result_list = []
        levels_map = dict()
         
        if self.pool is None:
            self.pool = ThreadPool(Calculator.ThreadPoolCount)
        legal_moves_dict = defaultdict(lambda: 0) 

        next_gen = self.calc_moves_score_worker(current_board.copy(), is_white, 1, depth, old_eval,
                       1, False, levels_map, [], result_list, legal_moves_dict)

        def process_generation(x):
            return [] if x is None else list(self.calc_moves_score_worker(*x))

        pv_moves = [] 
        if self.INCLUDE_PV:  # Only get PV if INCLUDE_PV is True
            future = self.pool.apply_async(self.get_pv, (current_board, depth, pv_moves))

        while True: 
            current_gen = next_gen
            next_gen = []
            for k in self.pool.imap_unordered(process_generation, current_gen, 1):
                next_gen += k
            if len(next_gen) == 0:
                break 

        if self.INCLUDE_PV:
            future.get()

        pv_moves_dict = {}
        if self.INCLUDE_PV:
            pv_moves_dict = {move_sequence: (score, 0) for move_sequence, score in pv_moves}

            for eval, _, sequence, _, _ in result_list: 
                sequence_str = ';'.join(sequence)
                for pv_sequence in pv_moves_dict:
                    if sequence_str.startswith(pv_sequence[:len(sequence_str)]):
                        pv_moves_dict[pv_sequence] = (pv_moves_dict[pv_sequence][0], 1)
                        break

        return result_list, levels_map, legal_moves_dict, pv_moves_dict

    def print_stats(self, curb, iswhite, full=True):
        #if neverthrow(asyncio.get_event_loop) is None:
        asyncio.set_event_loop(asyncio.new_event_loop())

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(self.async_print_stats(curb, iswhite, full))
        else:
            print(loop.run_until_complete(self.async_ret_stats(curb, iswhite, full)))


    async def async_ret_stats(self, curb, iswhite, full=True, **kwargs):
        with SimpleExceptionContext("async_ret_stats"):
            if not full:
                lev = self.get_score(curb, iswhite)
                mystr = f"score: {lev / 100}"
                return mystr
            else:
                return self.ret_stats(curb, iswhite, **kwargs)

    def ret_stats(self, curb, iswhite, format='plain'):
        result = self.calc_stability(curb, iswhite)
        return result.format_stats(format)


    async def async_print_stats(self, fen, iswhite, full=True):
        print(await self.async_ret_stats(fen, iswhite, full))

    @staticmethod
    def get_game(pgn):
        p=chess.pgn.read_game(io.StringIO(pgn))
        gam=p.game()
        return gam

    @simple_exception_handling("calc_entire")
    def calc_entire_game(self, pgn=None):
        def with_p(k, san, *args):
            tt = display_board(*args)
            return k, san, tt

        gam = Calculator.get_game(pgn)
        g = deepcopy(gam)

        ls = []
        k = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            while True:
                k += 1
                prev = g.board()
                g = g.next()
                if not g:
                    break
                b = g.board()
                san = prev.san(g.move)

                if k > self.MINMOVES:
                    f = executor.submit(display_board, b, g.turn, self.path)
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
    @timer.time_execution 
    def calc_stability(self, cur_board, iswhite):
        '''
        Calculates stability by first getting score of all reasonable moves, then applies calculation (see readme).
        Returns a StabilityStats object containing various stability metrics.
        '''
        r = StabilityStats(self.EXTENDED_STATS, cur_board.ply())
        
        # Get initial score (repeated 3 times for consistency)
        r.score = self.get_score(cur_board, iswhite)
        r.score = self.get_score(cur_board, iswhite)
        r.score = self.get_score(cur_board, iswhite)
        
        # Calculate scores for all reasonable moves
        arr, lev, legalmovesdic, r.pv = self.calc_moves_score(cur_board, iswhite, r.score)

        # Process level information
        nlev = {k: len(v) for k, v in lev.items()} if isinstance(lev[1], list) else lev.copy()
        maxlev = max(nlev.keys())

        # Calculate move fractions
        movfraq = {x: nlev[x]/legalmovesdic[x] for x in legalmovesdic if x in nlev}
        r.same_stab_frq = movfraq[maxlev] * 100 if len(movfraq) % 2 == 0 else movfraq[maxlev-1] * 100
        r.diff_stab_frq = movfraq[maxlev] * 100 if len(movfraq) % 2 == 1 else movfraq[maxlev-1] * 100

        # Check if position is tactical
        r.tactical = (movfraq[1] + movfraq[2]) / 2 < 1/30 or (nlev[1] + nlev[2] < 20)

        # Calculate stability for same and different color moves
        r.reasonable_moves = np.array([x[0] for x in arr], dtype="float64")
        same = [x[0] for x in arr if x[1] % 2 == 0]
        diff = [x[0] for x in arr if x[1] % 2 == 1]
        r.stability_same, fraq_a = self._calc_stability_array(same, iswhite, r.score)
        r.stability_diff, fraq_b = self._calc_stability_array(diff, not iswhite, r.score)
        r.fraq_method = fraq_b or fraq_a
        # Combine stabilities
        r.stability_all = self._combine_stabilities(r.stability_same, r.stability_diff, same, diff, arr)

        # Calculate additional statistics
        r.stdev = np.sqrt(np.mean((r.reasonable_moves - r.score) ** 2))
        r.mean = r.reasonable_moves.mean()
        r.max_score_of_reasonable = np.max(r.reasonable_moves)
        r.min_score_of_reasonable = np.min(r.reasonable_moves)

        # Set moves by depth
        r.moves_by_depth = nlev if len(str(dict(lev))) > 300 else lev

        # Log debug information
        logging.debug(f"stability_same:{r.stability_same}, stability_diff:{r.stability_diff}, "
                      f"stability_all:{r.stability_all}, stdev:{r.stdev}, "
                      f"tactical:{r.tactical}, nlev:{nlev}")
        
        return r

    def _calc_stability_array(self, arr, iswhite, init):
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
        return stab, b

    def _combine_stabilities(self, stabsame, stabdiff, same, diff, arr):
        if np.isnan(stabsame):
            return stabdiff
        elif np.isnan(stabdiff):
            return stabsame
        else:
            return (stabsame * len(same) + stabdiff * len(diff)) / len(arr)


