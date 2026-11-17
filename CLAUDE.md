# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All tools live in the project virtualenv (`.venv`, created with `python -m venv .venv`).

- Install: `.venv/bin/pip install -e ".[dev,docs]"` (dev = pytest, pytest-cov, mypy, ruff, numpy, matplotlib).
- Tests: `.venv/bin/python -m pytest`. Single test: `.venv/bin/python -m pytest tests/test_engine.py::test_converges_on_target`.
- Coverage: `.venv/bin/python -m pytest --cov` (branch coverage, fails under 90%).
- Types: `.venv/bin/mypy` (strict, checks `src/` only).
- Lint: `.venv/bin/ruff check src tests examples benchmarks`.
- Docs: `.venv/bin/mkdocs serve` (live preview), `.venv/bin/mkdocs build --strict` (what CI runs).
- Benchmark: `.venv/bin/python benchmarks/numpy_vs_python.py [length] [population] [generations]`.

CI (`.github/workflows/`) runs the same checks on Python 3.10–3.14, plus a job without numpy/matplotlib.

## Names

- Product name **GALib** (docs, prose, repository `carmelolg/GALib`, Pages URL `https://carmelolg.github.io/GALib/`: a project
  site's path is the repository name, so the repository must be called exactly `GALib`). Code stays lowercase:
  import package `carmelolg.galib`, CSS classes `galib-*`.
- Distribution name `io.github.carmelolg.galib`, import name `carmelolg.galib`. `io.*` cannot be an import name
  (stdlib `io`). `carmelolg` is a PEP 420 namespace package: never add `src/carmelolg/__init__.py`.
- Every source file starts with the SPDX copyright header (`SPDX-FileCopyrightText` / `SPDX-License-Identifier:
  Apache-2.0`); keep it on new files. `LICENSE` is the verbatim Apache-2.0 text, attribution goes in `NOTICE`.

## Architecture

- `core/problem.py`: `Problem[G]` is what users subclass. `fitness` and `fitness_batch` default to each other;
  `check_fitness_implemented` (called by the engine constructor) rejects a problem overriding neither.
  Default `crossover`/`mutate` delegate to `OnePoint` / `random_reset`.
- `core/engine.py`: `GeneticAlgorithm` is a template method. `run` = `initialize` + loop of `step` +
  `_record_generation` (stats, callbacks, stop conditions, generation 0 included). `step` = `reproduce` →
  `replace`. Operator precedence: constructor argument > `Problem` method > library default.
- `operators/`: `protocols.py` defines the positional-only call signatures; `selection` takes
  `(population, count, rng)`. Built-ins are classes with `__call__`. `_genome.py` holds the shared helpers:
  `rebuild`/`concat` (keep the genome type, raise `TypeError` for unsupported types), `numpy_rng`,
  `bernoulli_positions`, bounds handling.
- `metrics.py`: diversity measures (`DiversityMeasure = (genomes) -> float | None`), with numpy fast paths.
- `termination.py`, `callbacks.py`, `plot.py` (lazy matplotlib import).

## Documentation site

MkDocs Material, sources in `docs/`, config in `mkdocs.yml`; the API reference is generated from the docstrings by mkdocstrings (sphinx style, but write lists and emphasis in Markdown: a plain indented "list" renders as a run-on paragraph).

- Look and feel is one file, `docs/stylesheets/extra.css`: section 1 holds the colour tokens for light and dark (change the accent in those four lines), the rest is commented by section. System fonts only (`font: false`): no external font requests.
- Navigation is grouped in labelled sections (Getting started, User guide, Examples, API reference, Project), no top tabs, in `mkdocs.yml`. New pages must be added to `nav` or strict builds warn.
- The landing page (`docs/index.md`) is hand-built HTML plus `md_in_html`: hero, terminal (real output of a real run, keep it truthful) and card grids. Groups of four cards use the `galib-four` class. It is standalone: the front matter hides sidebar, table of contents and the previous / next cards (`hide: footer`), the header turns teal through `body:has(.galib-hero)` rules, and a `galib-links` row replaces the navigation. Keep it independent from the sectioned look of the other pages (it was a deliberate request).
- Logo: the "popolazione" concept (a ring of dots around an orange centre). One SVG per background in `docs/assets/`: `logo-light.svg` (neutral light pages), `logo-dark.svg` (dark theme), `logo-teal.svg` (teal header of the landing page and the mobile menu band; it is also `theme.logo`). The header and footer show them as CSS background images, chosen by theme in `extra.css` (they are multi-colour with gradients, so they cannot be recoloured with a CSS mask). `favicon.svg` is a single file whose colours follow the browser theme through `prefers-color-scheme`. The other concepts and the mono variants of the set (25 files) are not in the repository.
- Templates are overrides in `docs/overrides/` (`theme.custom_dir`, excluded from publishing with `exclude_docs`): `partials/footer.html` (brand, link columns, copyright bar) and `partials/pager.html`, the previous / next links as plain text at the end of the article, included by `partials/content.html` (a copy of Material's: re-sync it when upgrading Material). A page hides the pager with `hide: [footer]` (the landing page does). The footer links are written out by hand: update them when pages move.
- Layout on desktop (>= 76.25em): the primary sidebar is `position: fixed` and never moves; the main area and the footer leave room for it (footer only spans the content column), as in the JCAL docs. Do not give the sidebar a fixed height while it is sticky: Material's script shrinks the list near the footer and the list slides out of view.
- Vertical rhythm: every box (code block, note, tabs, table, diagram) has `1.2rem` above and below, set in the "vertical rhythm" rule of `extra.css`; margins of neighbouring boxes collapse, so two boxes in a row are 24px apart. Code blocks have no margin of their own elsewhere (`.highlight pre` is reset), so do not remove that rule or consecutive code blocks touch. Measure it rather than eyeballing: the gap between consecutive siblings of `.md-content__inner`.
- Material 9.8 specifics that bit before: Material gives the sidebar `align-self: flex-start`, which makes a `position: fixed` flex child ignore `top` and `bottom` and take its content height (reset it with `align-self: auto`); the card grid is the `.grid` element itself (`ul` is `display: contents`); Material's `[dir=ltr] .md-content__inner` margins need a higher-specificity selector to override; the permalink pilcrow in a big heading can wrap onto an invisible extra line.
- Check visually, not just with the strict build: `mkdocs build -d <dir>`, serve it with `python -m http.server`, then headless Firefox: `MOZ_HEADLESS=1 firefox --no-remote --profile <empty dir> --screenshot out.png --window-size 1280,1000 <url>` (the viewport is 15px narrower than the window). Test light and dark, and a 390px wide screen. Anything that depends on scrolling (fixed or sticky sidebars, footer, header over content) does NOT show in a static screenshot: drive Firefox through `geckodriver` (WebDriver over HTTP, no extra Python packages needed), scroll with `window.scrollTo`, wait about a second for Material's scroll handlers, then read `getBoundingClientRect()` and take a screenshot. Measure rather than guess: compare the sidebar box at scroll 0 and at the end of the page.

## Invariants to keep

- `Individual.score` is always "higher is better" (`-fitness` when `maximize` is False). Operators compare
  `score`, never `fitness`. `Individual` compares by identity (`eq=False`).
- The engine never relies on the population being sorted: use `max`/`min` by score.
- All randomness comes from the run's `random.Random` (`engine.rng`, passed to operators). Vectorised numpy code
  uses `numpy_rng(rng)`, a generator cached per `Random` instance, so runs stay reproducible. Never use the
  global `random` or `numpy.random` functions.
- Operators never modify their inputs and return the same genome type they received.
- numpy is optional: import it only through `operators/_genome.py` (`np` is `None` without numpy) and guard
  with `is_array`. Tests needing it use `pytest.importorskip("numpy")`.
- Fitness values must be finite; `evaluate` raises `ValueError` otherwise.

## Releases

- Releases are automated with release-please (`release-please-config.json`, `.release-please-manifest.json`,
  `.github/workflows/release-please.yml`). Commit messages must follow Conventional Commits (`feat:`, `fix:`,
  `docs:`, ...): the type decides the version bump.
- `CHANGELOG.md` is generated by release-please: never edit it by hand. The explanation of the process and the
  hand-written feature list of 0.1.0 live in `docs/changelog.md`, which includes `CHANGELOG.md`.
- The version lives in `pyproject.toml` and in `src/carmelolg/galib/__init__.py` (the line marked
  `x-release-please-version`); release-please updates both, so do not bump them manually.
- Nothing is published to PyPI.
