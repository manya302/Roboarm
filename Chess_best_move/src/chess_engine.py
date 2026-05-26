import chess
import chess.engine
import os
import shutil

def setup_stockfish():
    paths_to_check = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            return path

    path = shutil.which("stockfish")
    if path:
        return path

    raise FileNotFoundError(
        "Stockfish not found. Install it with:\n"
        "  Mac:   brew install stockfish\n"
        "  Linux: sudo apt install stockfish"
    )


def get_best_move(fen, time_limit=1.0):
    stockfish_path = setup_stockfish()

    try:
        board = chess.Board(fen)

        if not board.is_valid():
            status = board.status()
            errors = []

            status_checks = [
                ("STATUS_NO_BLACK_KING",     "black king missing"),
                ("STATUS_NO_WHITE_KING",     "white king missing"),
                ("STATUS_TOO_MANY_KINGS",    "too many kings"),
                ("STATUS_OPPOSITE_CHECK",    "the opposite side is in check — impossible position"),
                ("STATUS_TOO_MANY_CHECKERS", "too many pieces giving check"),
            ]
            for attr, message in status_checks:
                flag = getattr(chess, attr, None)
                if flag is not None and (status & flag):
                    errors.append(message)

            reason = ", ".join(errors) if errors else "invalid position"
            return None, None, f"Invalid position: {reason}"

        if board.is_game_over():
            outcome = board.outcome()
            return None, None, f"Game already over: {outcome.termination.name}"

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        engine.quit()

        if result.move is None:
            return None, None, "Stockfish returned no move"

        san_move = board.san(result.move)
        return result.move, san_move, None  # None = no error

    except chess.engine.EngineError as e:
        return None, None, f"Engine error: {e}"
    except Exception as e:
        return None, None, f"Unexpected error: {e}"