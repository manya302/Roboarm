import os
import sys
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import config and src modules
from src import config
from src.data_loader import load_image, load_ground_truth
from src.board_processor import detect_board_contour, get_perspective_transform, warp_board, draw_grid
from src.piece_detector import load_model, detect_pieces
from src.fen_generator import generate_fen
from src.chess_engine import get_best_move

app = Flask(__name__, template_folder='frontend', static_folder='frontend')

# Global variables to store model and state
model = None
current_image = None
current_image_path = None
current_image_file = None

def load_model_once():
    """Load model once at startup"""
    global model
    if model is None:
        print(f"Loading model from: {config.MODEL_PATH}")
        model = load_model(config.MODEL_PATH)
    return model

def get_human_readable_move(san_move):
    """Convert SAN move to human-readable description with check/checkmate indicators"""
    is_checkmate = san_move.endswith('#')
    is_check = san_move.endswith('+') and not is_checkmate
    
    clean_move = san_move.rstrip('#+')
    
    piece_map = {
        "Q": "Queen",
        "R": "Rook",
        "B": "Bishop",
        "N": "Knight",
        "K": "King"
    }
    
    if clean_move == "O-O":
        move_text = "King castles kingside"
    elif clean_move == "O-O-O":
        move_text = "King castles queenside"
    else:
        if clean_move[0] in piece_map:
            piece = piece_map[clean_move[0]]
            move_body = clean_move[1:]
        else:
            piece = "Pawn"
            move_body = clean_move
        
        if "x" in clean_move:
            square = clean_move.split("x")[-1]
            move_text = f"{piece} captures {square}"
        else:
            square = clean_move[-2:]
            move_text = f"{piece} moves to {square}"
    
    if is_checkmate:
        move_text += " — CHECKMATE!"
    elif is_check:
        move_text += " — CHECK!"
    
    return move_text

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string for HTML display"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/load_image', methods=['POST'])
def load_image_endpoint():
    """Load the chess board image"""
    global current_image, current_image_path, current_image_file
    
    try:
        # Load the image based on config
        current_image, current_image_path, current_image_file = load_image(
            mode=config.MODE,
            dataset_path=config.DATASET_PATH,
            fixed_image_name=config.FIXED_IMAGE_NAME
        )
        
        # Encode image for display
        img_base64 = encode_image_to_base64(current_image)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'image_file': current_image_file,
            'message': f'Image loaded: {current_image_file}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the chess position and find best move"""
    global current_image
    
    if current_image is None:
        return jsonify({
            'success': False,
            'error': 'No image loaded'
        })
    
    try:
        # Get turn from request
        data = request.json
        turn_color = data.get('turn', 'w')
        
        # Load model
        model = load_model_once()
        
        # Process board
        board_contour = detect_board_contour(current_image)
        M = get_perspective_transform(board_contour, config.BOARD_SIZE)
        warped = warp_board(current_image, M, config.BOARD_SIZE)
        
        # Draw grid
        grid, square_size = draw_grid(warped, config.BOARD_SIZE)
        
        # Detect pieces
        results, predicted_data = detect_pieces(
            model=model,
            image_path=current_image_path,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            M=M,
            board_size=config.BOARD_SIZE,
            square_size=square_size,
            files=config.FILES,
            ranks=config.RANKS
        )
        
        # Generate FEN with correct turn
        fen_without_turn = generate_fen(predicted_data, config.PIECE_MAP, config.FILES, config.RANKS)
        fen = fen_without_turn.split(" ")[0] + f" {turn_color} - - 0 1"
        
        # Get best move from Stockfish
        best_move, san_move, error = get_best_move(fen, time_limit=1.0)
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        if san_move:
            human_move = get_human_readable_move(san_move)
            
            # Create overlay image with detections
            plot_img = results[0].plot()
            plot_img_base64 = encode_image_to_base64(plot_img)
            
            return jsonify({
                'success': True,
                'fen': fen,
                'san_move': san_move,
                'human_move': human_move,
                'turn': 'White' if turn_color == 'w' else 'Black',
                'pieces_detected': len(predicted_data),
                'detection_overlay': plot_img_base64
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not calculate best move'
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Load model at startup
    load_model_once()
    print("\n" + "=" * 50)
    print("Chess Best Move Web Server")
    print("=" * 50)
    print(f"Server running at: http://localhost:8000")
    print("Press Ctrl+C to stop")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8000)