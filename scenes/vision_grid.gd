extends Node2D

const GRID_RADIUS_X = 8   # Half-width of 16
const GRID_RADIUS_Y = 11  # Half-height of 22
const GRID_SIZE = 16
const AI_MARKER = 3  # 👾 AI's position marker

func _ready():
	add_to_group("VisionGrid")

func get_vision_grid():
	var ai_players = get_tree().get_nodes_in_group("AIPlayer")
	if ai_players.is_empty():
		print("❌ ERROR: No AIPlayer found!")
		return []

	var ai_player = ai_players[0] as CharacterBody2D
	var player_pos = ai_player.global_position

	var vision_grid = []
	var tilemap_layer = find_tilemap_layer()
	if tilemap_layer == null:
		print("❌ ERROR: No valid TileMapLayer found!")
		return []

	var coins = get_tree().get_nodes_in_group("Coins")
	var ai_coords = Vector2i(player_pos.x / GRID_SIZE, player_pos.y / GRID_SIZE)

	# 🔥 FIXED: Adjusted range to prevent off-by-one error
	for y in range(-GRID_RADIUS_Y, GRID_RADIUS_Y):
		var row = []
		for x in range(-GRID_RADIUS_X, GRID_RADIUS_X):
			var check_position = player_pos + Vector2(x * GRID_SIZE, y * GRID_SIZE)
			var tile_coords = Vector2i(check_position.x / GRID_SIZE, check_position.y / GRID_SIZE)

			var tile_value = detect_tile(check_position, tilemap_layer, coins)

			# 👾 Mark AI position in the grid
			if tile_coords == ai_coords:
				tile_value = AI_MARKER  

			row.append(tile_value)
		vision_grid.append(row)

	# 🔥 FIXED: Added checks to validate grid size before returning
	if vision_grid.size() != 22:
		print("❌ ERROR: Vision Grid has", vision_grid.size(), "rows instead of 22!")
	if vision_grid[0].size() != 16:
		print("❌ ERROR: Vision Grid row has", vision_grid[0].size(), "columns instead of 16!")

	print("📌 AI Tile Coords:", ai_coords)
	print("🔍 Vision Grid Size:", len(vision_grid), "x", len(vision_grid[0]) if vision_grid else 0)

	return vision_grid

func detect_tile(position: Vector2, tilemap_layer, coins):
	if tilemap_layer == null:
		print("❌ ERROR: TileMap layer is NULL!")
		return 0

	var tile_coords = Vector2i(position.x / GRID_SIZE, position.y / GRID_SIZE)

	if tilemap_layer.get_used_rect().has_point(tile_coords):
		var source_id = tilemap_layer.get_cell_source_id(tile_coords)
		if source_id != null and source_id != -1:
			return 1  # 🧱 Obstacle

	# 🪙 Check if a coin exists at this position
	for coin in coins:
		if coin.global_position.distance_to(position) < GRID_SIZE / 2:
			return 2  # 🪙 Coin detected!

	return 0  # Empty space

func find_tilemap_layer():
	var tilemap_nodes = get_tree().get_nodes_in_group("TileMap")

	if tilemap_nodes.is_empty():
		print("❌ No TileMap Node2D found! Check if it's in the 'TileMap' group.")
		return null

	var tilemap_node = tilemap_nodes[0]  # Get the first TileMap Node2D

	print("✅ Found TileMap Node:", tilemap_node.name)

	# Loop through the children of the TileMap Node2D to find the TileMapLayer
	for child in tilemap_node.get_children():
		if child is TileMapLayer:
			print("🔍 Found TileMapLayer:", child.name)
			if child.name == "Mid":  # Ensure it's the correct layer
				print("✅ Using TileMapLayer:", child.name)
				return child

	print("❌ 'Mid' layer not found in TileMap!")
	return null
