extends Area2D

@export var player: CharacterBody2D
@export var ai_player: CharacterBody2D

func _on_body_entered(body):
	if body == player or body == ai_player:
		print("[DEBUG] Respawning:", body.name)
		if body.has_method("respawn"):
			body.respawn()
