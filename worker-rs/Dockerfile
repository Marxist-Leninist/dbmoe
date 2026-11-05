# The Rust worker in a container: one static binary on an empty base image (CPU training; for a GPU, run the native
# binary built with --features wgpu). Build from the repository root:
#
#   docker build -f worker-rs/Dockerfile -t dbmoe-worker-rs .
#   docker run --rm dbmoe-worker-rs parity --fixture /parity.json
#   docker run --rm -v dbmoe-worker:/data dbmoe-worker-rs worker --url URL --threads 4
#
# On Linux a container shares the host kernel, so CPU training runs at native speed; Docker Desktop on Windows and
# macOS runs containers in a Linux VM. ~/.dbmoe (the credential and worker label) lives in the /data volume.
FROM rust:1-alpine AS build
RUN apk add --no-cache musl-dev gcc
WORKDIR /src
COPY worker-rs/Cargo.toml worker-rs/Cargo.lock worker-rs/
COPY worker-rs/src worker-rs/src
COPY worker-rs/tests worker-rs/tests
RUN cargo build --release --locked --manifest-path worker-rs/Cargo.toml

FROM scratch
COPY --from=build /src/worker-rs/target/release/dbmoe-worker /dbmoe-worker
COPY apple/parity.json /parity.json
ENV HOME=/data
VOLUME /data
ENTRYPOINT ["/dbmoe-worker"]
CMD ["help"]
