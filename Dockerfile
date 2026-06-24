# syntax=docker/dockerfile:1

# ---- build a static metis binary (musl, pure-Rust http — no openssl/cgo/ring) ----
FROM rust:1.89-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --locked --release --bin metis

# ---- minimal runtime image ----
FROM alpine:3.20
RUN apk add --no-cache ca-certificates \
 && adduser -D -u 10001 metis
WORKDIR /app
COPY --from=build /src/target/release/metis /usr/local/bin/metis
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
