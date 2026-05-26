def generate_fen(predicted_data, piece_map, files, ranks):
    """
    Convert predictions to FEN notation
    """
    # Create empty board
    board = [["" for _ in range(8)] for _ in range(8)]
    
    # Fill board with pieces
    for class_name, square in predicted_data:
        file = square[0]
        rank = square[1]
        
        col = ord(file) - ord('a')
        row = int(rank) - 1
        
        class_name_fixed = class_name.replace("-", "_")
        
        if class_name_fixed in piece_map:
            board[row][col] = piece_map[class_name_fixed]
        else:
            print(f"Unknown class: {class_name}")
    
    # Convert board to FEN
    fen_rows = []
    
    for r in range(7, -1, -1):  # rank 8 to 1
        empty = 0
        fen_row = ""
        
        for c in range(8):
            piece = board[r][c]
            
            if piece == "":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece
        
        if empty > 0:
            fen_row += str(empty)
        
        fen_rows.append(fen_row)
    
    fen = "/".join(fen_rows) + " w - - 0 1"
    return fen