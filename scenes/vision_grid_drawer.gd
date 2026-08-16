extends Node2D

const GRID_RADIUS = 3
const GRID_SIZE = 16
const COLORS = {
	0: Color(1, 1, 1, 0.2),  # Empty space (light white)
	1: Color.RED,            # Obstacles
	2: Color.YELLOW          # Coins
}

var vision_grid = []
var player_pos = Vector2.ZERO

func _ready():
	queue_redraw()

func set_grid(grid_data, player_position):
	vision_grid = grid_data
	player_pos = player_position
	queue_redraw()

func _draw():
	if vision_grid.is_empty():
		return

	for y in range(vision_grid.size()):
		for x in range(vision_grid[y].size()):
			var cell_value = vision_grid[y][x]
			var cell_color = COLORS.get(cell_value, Color(1, 1, 1, 0.5))  # Default white
			var cell_position = player_pos + Vector2(
				(x - GRID_RADIUS) * GRID_SIZE, 
				(y - GRID_RADIUS) * GRID_SIZE
			)

			draw_rect(Rect2(cell_position - Vector2(GRID_SIZE / 2, GRID_SIZE / 2), 
				Vector2(GRID_SIZE, GRID_SIZE)), cell_color, true)
