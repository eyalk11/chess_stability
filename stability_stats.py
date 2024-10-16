from tabulate import tabulate


class StabilityStats:
    def __init__ (self,extended_stat=False,half_move_number=2):
        self.same_stab_frq=0
        self.diff_stab_frq=0
        self.score = 0
        self.stability_all = 0
        self.stability_same = 0
        self.stability_diff = 0
        self.reasonable_moves = []
        self.max_score_of_reasonable = []
        self.min_score_of_reasonable = []
        self.mean = 0
        self.stdev = 0
        self.tactical = False
        self.moves_by_depth = []
        self.pv= []
        self.extended_stat=extended_stat
        self.half_move_number=half_move_number
        self.fraq_method=False #fraction method

    def assign(self, score, stability_all, stability_same, stability_diff, num_of_reasonable_moves, max_score_of_reasonable,
                 min_score_of_reasonable, mean, stdev, tactical, moves_by_depth):
        self.score = score
        self.stability_all = stability_all
        self.stability_same = stability_same
        self.stability_diff = stability_diff
        self.reasonable_moves = num_of_reasonable_moves
        self.max_score_of_reasonable = max_score_of_reasonable
        self.min_score_of_reasonable = min_score_of_reasonable
        self.mean = mean
        self.stdev = stdev
        self.tactical = tactical
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
            'num of reasonable moves': len(self.reasonable_moves),
            'max(score) of reasonable': self.max_score_of_reasonable,
            'min(score) of reasonable': self.min_score_of_reasonable,
            'mean': self.mean,
            'stdev': self.stdev,
            'fraction method': self.fraq_method,
            'moves by depth': self.moves_by_depth
        }
        if not self.extended_stat:
            data = {k: v for k, v in data.items() if k in minimal_keys}

        result_str = f"Score: {dic_formats['score'].format(data['score'])}\n"
        result_str += "Few available moves\n" if self.tactical else ""


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