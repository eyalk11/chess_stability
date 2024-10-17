import asyncio
import websockets
import json
import chess
from calc import Calculator
Calculator.EXTENDED_STATS=False 
Calculator.INCLUDE_PV=False
Calculator.IGNORE_CONFIG=True
import sys
if sys.platform == 'win32':
    STOCKFISHPATH = r'c:/gitproj/stockfish/stockfish-windows-x86-64-avx2.exe'
else:
    STOCKFISHPATH = '/home/jovyan/stockfish/src/stockfish'
class ChessWebSocketServer:
    def __init__(self):
       
        self.calculator = Calculator(STOCKFISHPATH)
        self.current_task = None

    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
            if data["type"] == "FROM_CONTENT":
                print('got message')
                node_list = data["payload"]["nodeList"]
                board = chess.Board()
                
                for node in node_list[1:]:  # Skip the initial position
                    move = chess.Move.from_uci(node["uci"])
                    board.push(move)

                # Cancel the previous task if it's still running
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                
                # Create and start a new task
                self.current_task = asyncio.create_task(self.calculator.async_ret_stats(board, board.turn, format='html'))
                
                try:
                    text = await self.current_task
                    self.calculator.printtimer()
                    print('got result')
                    print(text)
                    await websocket.send(text)
                except asyncio.CancelledError:
                    print('Task was cancelled')

        except json.JSONDecodeError:
            print(f"Received invalid JSON: {message}")
        except KeyError:
            print(f"Received message with unexpected format: {message}")
        except chess.IllegalMoveError:
            print(f"Encountered illegal move in the sequence")

    async def listener(self, websocket, path):
        async for message in websocket:
            await self.handle_message(websocket, message)

    def run(self):
        start_server = websockets.serve(self.listener, "localhost", 8765)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    server = ChessWebSocketServer()
    server.run()
