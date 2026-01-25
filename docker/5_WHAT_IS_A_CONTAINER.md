# What is a container?

## The problem containers solve

A modern app often needs multiple components (for example: a frontend, an API, and a database). Without containers, you end up installing and maintaining many runtimes and services locally:

- Matching the same versions as teammates/CI/production is hard.
- Dependencies can conflict with what’s already installed on your machine.
- Upgrading one component can accidentally break another.

Containers help by packaging and running each component in a controlled, repeatable environment.

## Definition (practical)

A **container** is an isolated process with:

- its own filesystem view (from an image)
- its own network namespace (ports, IP, DNS inside Docker networks)
- its own resource limits (CPU/memory constraints you can configure)

Containers are created from **images** (read-only templates). A running container is an image + runtime configuration (ports, env vars, mounts, etc.).

### What makes containers useful

- **Self-contained**: includes the app + dependencies needed to run.
- **Isolated**: reduced interference with the host and other containers.
- **Independent**: you can start/stop/remove one without deleting the others.
- **Portable**: the same image can run anywhere with a compatible container runtime.

## Containers vs virtual machines (VMs)

- **VM**: runs a full guest OS with its own kernel (more overhead, slower to boot, heavier resource usage).
- **Container**: runs as a process on the host, sharing the host kernel (lighter, faster startup, higher density).

In practice, you’ll often see both together: cloud “machines” are frequently VMs, and those VMs run many containers.

## Try it: run a container (CLI)

Run a demo web container and publish it on your machine:

```console
$ docker run -d -p 8080:80 docker/welcome-to-docker
```

What the flags mean:

- `-d` runs in the background (detached)
- `-p 8080:80` maps host port `8080` → container port `80`

Open:

- `http://localhost:8080`

### View and stop it

```console
$ docker ps
$ docker stop <container_id_or_name>
```

Tips:

- `docker ps` shows only running containers; use `docker ps -a` to include stopped ones.
- You don’t need the full ID to stop a container—just enough characters to be unique.

## Try it: run a container (Docker Desktop GUI)

If you prefer the UI:

1. Open Docker Desktop and search for the image `docker/welcome-to-docker`.
2. Pull it (download the image).
3. Run it with a host port mapping (for example host `8080` → container `80`).
4. Use the **Containers** view to see logs, inspect settings, and stop/restart the container.

## Key idea: port publishing

Containers are isolated. Publishing a port (`-p`) is how you intentionally expose a container service to your host network:

- Without `-p`, a web server inside a container is not reachable from your browser on the host.
- With `-p host:container`, traffic to the host port is forwarded to the container port.
