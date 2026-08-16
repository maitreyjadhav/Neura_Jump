extends CharacterBody2D

const SPEED = 130.0
const JUMP_VELOCITY = -300.0
const DEATH_Y = 1000  # Y-position threshold for respawning

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

@onready var animated_sprite = $AnimatedSprite2D

# Store initial position for respawn
var START_X
var START_Y

func _ready():
	START_X = global_position.x
	START_Y = global_position.y

func _physics_process(delta):
	# Check if player falls below threshold
	if global_position.y > DEATH_Y:
		respawn()
		return  # Prevent further processing this frame

	# Apply gravity
	if not is_on_floor():
		velocity.y += gravity * delta

	# Handle jump
	if Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Get input direction (-1, 0, 1)
	var direction = Input.get_axis("move_left", "move_right")

	# Flip the sprite
	if direction > 0:
		animated_sprite.flip_h = false
	elif direction < 0:
		animated_sprite.flip_h = true

	# Play animations
	if is_on_floor():
		if direction == 0:
			animated_sprite.play("idle")
		else:
			animated_sprite.play("run")
	else:
		animated_sprite.play("jump")

	# Apply movement
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)

	move_and_slide()

func respawn():
	print("[DEBUG] Respawning Human Player...")
	global_position = Vector2(START_X, START_Y)
	velocity = Vector2.ZERO
