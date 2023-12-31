[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eyalk11/chess_stability/master?labpath=chessgame.ipynb)
# Chess Stability
Provide a new statistical tool for evaluting chess postions (in addition to standard score).
This tool is stability score (0-100). And describes how easy it is to break position / how stable the position is.   

## Usage

### Installation 

1. Download [stockfish](https://stockfishchess.org/download/)
2. Do:
```
****
pip install jupter
pip install python-chess
git clone git@github.com:eyalk11/chess_stability.git
```
3. Update `STOCKFISHPATH` in the notebook. 

### Running 

```
cd chess_stability 
jupyter lab -> select notebook
```

You can view your own games by adding a cell `display_game(pgn)`

(This shit is  interactive!) 

or evaluate a position by `display_board(fen,iswhite)`.

By default, takes `iswhite` from `fen`. 

See examples.

## Method

We define reasonable move (or not blunder) , as a move that does less damage than 200 centipawns.

We define a function $D(x)$ that returns $0$ if $x \ge 0$ and it is white's turn or $x \le 0$ and it is black turn. Otherwise, it is the identity. 
This function is meant to filter out improving moves. 

$C_1$ and $C_2$ some constants. 

The method is the following: 

1. Evaluate all reasonable moves of certain depth (for both players) $R$
2. Evalute current position $\mu$ 
3. If the position is worse than 200 centipawns (to each direction):

$$V= \sum_{r \in R } e^{-C_1 (D(eval(r)-\mu))^2} $$

4. Otherwise:

$$V= \sum_{r \in R } e^{-C_2 (\frac{ D(eval(r)-\mu)}{\mu})^2} $$

5. Denote stability by $\frac{V}{|R|}$

This is a twist on gaussian distrubtion, taking current position as $\mu$. 


## Notice

It is licensed under [by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Contact me for questions

eyalk@gmail.com 

Eyal Karni

   
   
