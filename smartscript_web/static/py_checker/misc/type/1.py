import chess
import chess.pgn
from board import ChessBoard
from mcts import MCTSAgent
from model import build_model
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


model = build_model()
model.load_weights("weights/zmodel")

window = ChessBoard()
window.board.push_san("e4")
window.board.push_san("c5")

while not window.board.is_game_over():
    window.draw()
    _, _, move = MCTSAgent.move(None, model, window.board.turn, window.board, 7.0, lambda: window.handle_events())
    print("==> " + window.board.san(move))
    window.board.push(move)

pgn = chess.pgn.Game.from_board(window.board)
print(pgn)