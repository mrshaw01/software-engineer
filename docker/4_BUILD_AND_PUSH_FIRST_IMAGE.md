# Build and push your first image

## What you’re trying to achieve

- Turn an application into a **container image** (a portable build artifact).
- Run that image locally to verify it works.
- Push it to a **registry** (Docker Hub or a private registry) so other machines/CI can pull it.

## Core concepts (the minimum you need)

- **Dockerfile**: a recipe for building an image.
- **Build context**: the directory you run `docker build` from (`.`). Docker sends this context to the builder, so keep it small.
- **Image tag**: a name + version-like label, e.g. `username/app:1.0.0`. Tags are just pointers; you can retag the same image.
- **Registry / repository**:
  - Registry: the server (for example `docker.io`, `ghcr.io`).
  - Repository: a named collection of images (for example `username/getting-started-todo-app`).

## Before you build (good habits)

### Use a `.dockerignore`

Exclude things that shouldn’t be sent to the builder (faster builds, smaller context):

- `node_modules/`, build artifacts, `.git/`, local secrets, IDE files

### Pick trusted base images

- Prefer official or well-maintained images.
- Keep them updated (security fixes come through base image updates).

## Build an image

From a directory that contains a `Dockerfile`:

```console
$ docker build -t <DOCKER_USERNAME>/getting-started-todo-app:latest .
```

Verify it exists:

```console
$ docker image ls
```

Run it locally (ports depend on the app; this is a common pattern):

```console
$ docker run --rm -p 8080:80 <DOCKER_USERNAME>/getting-started-todo-app:latest
```

## Tagging (don’t rely on only `latest`)

Use an explicit version tag too:

```console
$ docker tag <DOCKER_USERNAME>/getting-started-todo-app:latest <DOCKER_USERNAME>/getting-started-todo-app:1.0.0
```

Why:

- Makes rollbacks easier.
- Makes CI/CD and releases reproducible.

## Push to Docker Hub

### 1) Sign in

```console
$ docker login
```

### 2) Push a tag

```console
$ docker push <DOCKER_USERNAME>/getting-started-todo-app:latest
$ docker push <DOCKER_USERNAME>/getting-started-todo-app:1.0.0
```

If your repo is private, other users/machines need permission + authentication to pull.

## Registry mental model (useful when you move beyond Docker Hub)

An image reference can include a registry host:

- Docker Hub default: `username/app:tag` (implicitly `docker.io`)
- Explicit form: `docker.io/username/app:tag`
- GitHub Container Registry example: `ghcr.io/owner/app:tag`

## Quick troubleshooting

- **Push denied**: wrong repo name, not logged in, or no permission to that namespace.
- **Build is slow**: large build context; add `.dockerignore`, reduce copied files, leverage build cache.
- **Image is huge**: use multi-stage builds, minimize dependencies, use slimmer base images.
- **Works locally, fails elsewhere**: avoid copying local config/secrets into images; rely on env vars and runtime secrets.

## Practical best practices (keep these in mind)

- Build images to be **immutable**: configuration should be injected at runtime (env vars, mounts, secrets).
- Use **multi-stage builds** when compiling assets/binaries to keep runtime images small.
- Prefer **non-root** containers when feasible (depends on base image/app).
- Scan and update dependencies; treat images like any other artifact you patch and redeploy.
