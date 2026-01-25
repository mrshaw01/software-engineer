# Develop with containers

## What you’re doing (goal)

Now that Docker Desktop works, you can use containers as your **development environment**:

- Start all dependencies (DB, cache, backend, frontend) with one command.
- Avoid installing runtimes locally (Node, MySQL, etc.).
- Edit code on your machine and see changes reflected quickly via **bind mounts** + dev servers / watchers.

## The common pattern: Docker Compose for dev

In real projects, a `compose.yaml` typically defines multiple **services**, plus:

- **Ports** (how you reach a service from your host)
- **Volumes / bind mounts** (how source code or data is stored)
- **Environment variables** (configuration)
- **Networks** (how services talk to each other)

Compose is the “one file” that documents how to run the whole system locally.

## Try it out with a sample project

This example uses Docker’s “getting-started” to-do app (good for practice).

### 1) Get the code

```console
$ git clone https://github.com/docker/getting-started-todo-app
$ cd getting-started-todo-app
```

### 2) Start the dev environment

If the project provides a Compose setup, start it with one of these commands:

```console
$ docker compose up
```

Some projects also support a “watch” mode that syncs changes and restarts services when needed:

```console
$ docker compose watch
```

Then open:

- `http://localhost`

If it takes a while the first time, that’s normal: images may need to download and dependencies may need to initialize.

## What’s usually running (typical multi-container dev stack)

You’ll often see containers like:

- **Frontend** (React/Vite dev server)
- **Backend API** (Node/Go/Python service)
- **Database** (MySQL/Postgres)
- **Admin UI** (phpMyAdmin / Adminer)
- **Reverse proxy** (Traefik/Nginx) routing host requests to services

The big win: you don’t install these directly on your machine — Docker runs them.

## Why changes can show up “immediately”

Two mechanisms are common in dev setups:

- **Bind mounts**: your project folder on the host is mounted into the container, so the container sees your latest files.
- **Watchers / hot reload**: a process inside the container watches file changes and reloads (Vite, nodemon, etc.).

If hot reload doesn’t work, it’s usually because mounts or file watching aren’t set up correctly.

## Basic commands while developing

See what’s running:

```console
$ docker compose ps
```

Tail logs:

```console
$ docker compose logs -f
```

Open a shell in a service:

```console
$ docker compose exec <service_name> sh
```

Stop everything:

```console
$ docker compose down
```

## Small “edit something” exercise (what to focus on)

When following tutorials that have you edit backend/frontend files, focus on the idea:

- You change a file in your editor.
- The change is visible because the container sees your file (mount) and the app reloads (watcher).

You don’t need to memorize the demo’s exact files; the skill is understanding _why_ the feedback loop works.

## Troubleshooting quick hits

- **Port conflicts**: if `80`/`3000`/`8080` is taken, change host port mappings in `compose.yaml`.
- **No reload**: ensure the code directory is mounted; for Windows, keeping project files inside WSL can improve watcher behavior.
- **Can’t connect to DB**: services talk to each other via the Compose network using service names (not `localhost` inside containers).
- **Disk space**: images and volumes can grow; prune carefully once you understand what you’re deleting.
