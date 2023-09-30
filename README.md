How to use this repository
==========================

## Setting up

### Option 2: Use the Astro CLI

Download the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) to run Airflow locally in Docker. `astro` is the only package you will need to install.

1. Run `git clone https://github.com/TJaniF/astronomer-codespaces-test.git` on your computer to create a local clone of this repository.
2. Install the Astro CLI by following the steps in the [Astro CLI documentation](https://docs.astronomer.io/astro/cli/install-cli). The main prerequisite is Docker Desktop/Docker Engine but no Docker knowledge is needed to run Airflow with the Astro CLI.
3. Run `astro dev start` in your cloned repository.
4. After your Astro project has started. View the Airflow UI at `localhost:8080`.

#### Pre-commit

After cloning the repo, you should run the following command in order to add the pre-commit hook to
git:

```
pre-commit install
```

If you don't have pre-commit installed you can do it by running:

```
pip install pre-commit
```

Now, whenever you try making a commit, the pre-commit hook will run automatically before that. If
every step passes, the commit will proceed as usual. If any of the pre-commit steps fails, the
commit won't be executed.

In the case where the failure was just because there was a formatting problem that got fixed
automatically during the pre-commit, you can try committing again and the commit should proceed as
usual.

But if there was a step that required manual changes, you must fix them before being able to
continue with the commit.

If you want to just execute the pre-commit steps, you can run the command:

```
pre-commit run --all-files
```
