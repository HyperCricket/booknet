build:
	docker compose up
	docker build -t booknet .

run: build
	docker run booknet
