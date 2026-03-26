.PHONY: run

run:
	@test -n "$(IMAGE)" || (echo "Usage: make run IMAGE=./input/S1.jpg" && exit 1)
	uv run python im2oil.py run $(IMAGE) --palette-size 8 --num-brushes 4
