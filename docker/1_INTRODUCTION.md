# Docker introduction (and how to get it)

## What Docker is (in practice)

Docker is a toolchain for building and running applications as **containers**:

- **Image**: an immutable “package” that includes your app + runtime + dependencies (built from a `Dockerfile`).
- **Container**: a running instance of an image (a process with isolation + its own filesystem view).
- **Registry**: where images live (Docker Hub is the default public registry, but you can use private registries too).

Why people use it:

- **Consistency**: fewer “works on my machine” issues.
- **Faster delivery**: build once → test → deploy the same artifact.
- **Portability**: run the same container on laptops, servers, and CI (with a compatible container runtime).

## Mental model: client + daemon

Most workflows look like:

1. You run a command with the `docker` CLI (for example `docker build`, `docker run`).
2. The CLI talks to the Docker Engine (the daemon, `dockerd`).
3. The daemon builds images, runs containers, and manages networks/volumes.

`docker compose` is a companion tool that defines and runs multiple services together (usually via `compose.yaml`).

## “Get Docker”: Desktop vs Engine

The right install depends on your OS and what you need.

### macOS / Windows (typical path: Docker Desktop)

On macOS and Windows, you usually install **Docker Desktop**. It bundles:

- Docker Engine + CLI
- Docker Compose
- A local Linux VM (because most containers are Linux containers)
- UI + integrations (credentials, contexts, optional Kubernetes, etc.)

Notes:

- **Licensing**: Docker Desktop has licensing terms for some commercial use. Check Docker’s current pricing/licensing page before using it in a company setting.
- **Windows containers vs Linux containers**: Windows can run Windows containers, but most tutorials assume Linux containers. Desktop can switch modes, but compatibility differs.

### Linux (typical path: Docker Engine)

On Linux you typically install **Docker Engine** (daemon + CLI). Compose is often installed as a Docker plugin (`docker compose`).

Common post-install steps:

- Confirm the daemon is running (systemd service on many distros).
- Configure permissions if you want to run Docker without `sudo` (often by adding your user to the `docker` group).

## Quick verification checklist

After installation, these are useful sanity checks:

```console
$ docker version
$ docker info
```

Run a small test container:

```console
$ docker run --rm hello-world
```

If that works, your basics (CLI ↔ daemon ↔ registry pull ↔ container run) are in place.

## A few practical tips

- Prefer **small base images** (for example Alpine or distroless when appropriate) to reduce size/attack surface.
- Treat images as **build artifacts**: tag versions, avoid “latest-only” habits, and keep Dockerfiles reproducible.
- Use **volumes/bind mounts** intentionally:
  - bind mounts are great for local development (edit files on host, run inside container)
  - volumes are great for persistent data managed by Docker

## References

- Docker overview: <https://docs.docker.com/get-started/docker-overview/>
- Get Docker: <https://docs.docker.com/get-started/get-docker/>
