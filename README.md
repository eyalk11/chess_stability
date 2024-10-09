[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eyalk11/chess_stability/master?labpath=chessgame.ipynb)

# Chess Stability

Provide a new statistical tool for evaluation of chess positions (in addition to standard score).
This tool is stability score (0-100). And describes how easy it is to break position / how stable the position is.   

## Introduction 


For half a century, computers have been utilized to analyze chess positions.
Throughout this period, a single measure  was established as the only numerical metric[^1] which is used in computerized analysis. I refer to the infamous **centipawn loss** of course, that signifies how good the position is (in terms of pawn advantage). 

This is the case even though this method fails to account for some very important aspects of the position. For instance, there could be a position that is only good because of 10-moves line no human could find. The evalution might be misleading in this case.  Or the position could look calm and solid, sometimes even winning, but one has to play very precisely in order to maintain the advantage. While analysing a position, a player should be able to discern whether it is his opponent who would struggle to find good moves down the line, or he is.  Ideally, that information should be readily available to him. 

In all the cases montioned, one could argue that the position is not stable. Indeed, quite often **keeping the position stable is even more important than playing the absolutely best move**. Or rather, the stability of the position matters more than the absoulte evalution[^2]. That is of course, from human perspective, as we are very bad at playing positions computers excel in. Even the best players tend to make mistakes when the number of good options is limited.  That is why I have developed an algorithm that evaluates the stability of chess positions.

We have already discussed the intiution behind stability. However, this notion could have a precise mathematical sense, as defined by the algortihm[^3], but it still a bit hard to articulate. So, it is roughly:   *The expected percentage change in evalution resulting from playing a random reasonable line of a certain depth[^4]*. Another candidate(not implemented yet) could be *percentage of reasonable lines of certain depth that maintain a relatively consistent evaluation*(in other words, approved by the computer). 

But what are reasonable moves?  Those are in prinicple human moves, or intuitive moves. It should come as no surprise that computers are perfectly capable of playing human-like moves , as the bots in chess.com keep us entertained. The easiest way to achieve it is to  limit their time to think. 

The challenge here lies in devising the right algorithm or formula to calculate this stability, make it perform well, and also in choosing the right parameters. While I am quite satisficed with the current result, it is not perfect yet. But in principle, I think this should demonstrate that such measure is both useful (even in its current state), and of course achievable (could be implemented). And I find it strange, it hasn't been devised before[^5]. 

 The version of the  algorithm[^3] presented here is rather crude, because of some dubious optimizations that had to be introduced in the code. 



## Usage

**You can just click on binder link to see running interative version of notebook** 

It is a bit slow, so it is better to run it locally. 

### Installation 

1. Download [stockfish](https://stockfishchess.org/download/)
2. Do:
```
****
pip install jupter
pip install python-chess
git clone git@github.com:eyalk11/chess_stability.git
```
### Running 

```
cd chess_stability 
jupyter lab 
```
4. Select jupyter notebook called `chessgame` .


5. (Once) update  `STOCKFISHPATH` path in the notebook. 

---

There are some example games in this notebook,  which you can interact with. It displays the stability alongside other measures below the board (asynchronously).   

You can view your own games by adding a cell `display_game(pgn)`

or evaluate a position by `display_board(fen)`.


## Method

Suppose we have a certian position. We define a reasonable move (or not blunder) , as a move that does less damage than $P_1$ centipawns. 
We define a winning position if it is better than $P_2$ centipawns relatively to the playing player. 
Let $C_1$ and $C_2$ some constants. 

We define function $D(x)$:  

$$
D_{turn}(x) = 
\begin{cases} 
0 & \text{if } x \ge 0 \text{ and it's White's turn} \\
0 & \text{if } x \le 0 \text{ and it's Black's turn} \\
x & \text { otherwise} 
\end{cases}
$$

This function is meant to filter out improving moves. 


The method is the following: 

1. Evaluate current position $\mu$ 
2. Evaluate all lines of certain depth. We usually use weaker engine to calculate those. Defined the set of candidate lines as $L$.
3. Filter out  moves that does more than $P_1$ damage. 
5. If the position is winning, use: 

$$V:= \sum_{l \in L } e^{-C_2 \left(\frac{ D_{turn(l)}(eval(l)-\mu)}{\mu}\right)^2} $$


4. Otherwise:

$$V:= \sum_{l \in L } e^{-C_1 \left(D_{turn(l)}(eval(l)-\mu)\right)^2} $$

5. Denote stability by $\frac{V}{|L|}$

This is a twist on gaussian distribution.


In the current implementation , there are actually three flavours of stability:  total, same color and different color stability. In case the final evaluted position is of the same color as the initial one, it is taken into account in the same color stability. Otherwise, it is part of the different color stability. The weighted average is the total stability.  

## Notice

It is licensed under [by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Contact me for questions

eyalk@gmail.com 

Eyal Karni

[^1]: People also look at the primary variations yield by the computer.
[^2]: Some players would prefer it to be stable, some unstable and perheps tactical. 
[^3]: Presented in Method section, of course.
[^4]: Depth is the number of moves in the line. We typicaly keep it low because of performance considerations. 
[^5]: The notion of stability plays a key role in physics and other fields. 
