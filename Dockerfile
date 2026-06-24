# syntax=docker/dockerfile:1

# ---- build a static metis binary (no cgo, pure stdlib) ----
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/metis ./cmd/metis

# ---- minimal runtime image (~15 MB) ----
FROM alpine:3.20
RUN apk add --no-cache ca-certificates \
 && adduser -D -u 10001 metis
WORKDIR /app
COPY --from=build /out/metis /usr/local/bin/metis
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && mkdir -p /app/library /app/docs \
 && chown -R metis:metis /app
USER metis

# The model RAM lives in the ollama service; metis itself is tiny.
ENV OLLAMA_HOST=http://ollama:11434 \
    METIS_MODEL=qwen3:1.7b \
    PORT=8080
EXPOSE 8080

# entrypoint waits for ollama + pulls the Cortex & embedder, then runs the command (default: serve)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
