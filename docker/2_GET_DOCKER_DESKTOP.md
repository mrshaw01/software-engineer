# Get Docker Desktop

## What Docker Desktop is

Docker Desktop is the easiest way to get a working Docker environment on **macOS** and **Windows**. It bundles the Docker CLI plus a managed Docker Engine and adds a GUI to manage containers, images, volumes, and settings.

Why it exists: most containers are **Linux containers**. On macOS/Windows, Docker Desktop runs a small Linux VM under the hood so you can run Linux containers locally.

## Licensing note (important)

Docker Desktop has licensing terms for some commercial use (for example, larger enterprises). Before using it at work, check Docker’s current licensing/pricing page.

## Install checklist

1. Download Docker Desktop for your OS and install it.
2. Start Docker Desktop and finish the initial setup.
3. Confirm the engine is running:

```console
$ docker version
$ docker info
```

### Windows-specific tips

- Prefer the **WSL 2** backend (it’s the most common/recommended setup for Linux containers on Windows).
- If something feels “slow”, check Docker Desktop resource settings (CPU/RAM/disk) and whether you’re storing project files inside WSL vs on the Windows filesystem.

## Run your first container

Run a detached container that publishes port `8080` on your machine to port `80` in the container:

```console
$ docker run -d -p 8080:80 docker/welcome-to-docker
```

Then open:

- `http://localhost:8080`

To stop and clean up later:

```console
$ docker ps
$ docker stop <container_id>
```

(Or use `--rm` for short-lived containers when you don’t need to keep them.)

## Manage it in Docker Desktop

In Docker Desktop you can:

- View running containers (start/stop/restart, see logs).
- Inspect details (ports, env vars, mounts, networks).
- Open a shell inside a container (often called **Exec**).
- Watch resource usage (CPU/memory) and troubleshoot.

## Common gotchas

- “Docker isn’t running”: Docker Desktop must be started; the CLI talks to the engine it provides.
- Port already in use: if `8080` is taken, map a different host port (for example `-p 8081:80`).
- Images vs containers: deleting a container doesn’t delete the image; images take disk space.
