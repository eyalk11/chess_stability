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
3. Update STOCKFISHPATH in the notebook. 

### Running 

```
cd chess_stability 
jupyter lab -> select notebook
```

## Method

We define reasonable move (or not blunder) , as a move that does less damage than 200 centipawns.

The method is the following: 

1. Evaluate all reasonable moves of certain depth (for both players) $R$
2. Evalute current position $\mu$ 
3. If the position is worse than 200 centipawns (to each direction):

$$ \sum_{r \in R } e^{-C (eval(r)-\mu)^2} $$

4. Otherwise:

$$ \sum_{r \in R } e^{-C (\frac{eval(r)-\mu}{\mu}^2}) $$

## Notice

It is licensed under [by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Contact me for question eyalk@gmail.com 

Eyal Karni

   
   
