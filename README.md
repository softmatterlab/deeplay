````markdown
# Documentation Setup and Automation Guide

This guide explains how to set up and maintain documentation for your Python library in a dedicated `docs` branch with GitHub Pages. It also details how all the steps, once configured, are automated via a GitHub Actions workflow that updates and builds documentation upon new releases.

## Overview

1. **Dedicated `docs` branch**: Documentation is served from the `docs` folder on the `docs` branch, utilizing GitHub Pages.
2. **Automated `.rst` File Generation**: A Python script (`generate_doc_markdown.py`) scans your codebase and generates `.rst` files automatically.
3. **Sphinx & Extensions**: Use Sphinx, `sphinx-automodapi`, and `pydata-sphinx-theme` to build the docs.
4. **CI/CD with GitHub Actions**: Once set up, the provided GitHub Actions workflow automatically updates the documentation every time a new release is published.

## Initial Setup Steps

### 1. Create the `docs` Branch on GitHub

- Go to the repository’s **Settings**.
- Under **Pages**, configure the site to be served from the `docs/` folder on the `docs` branch.
- GitHub will create the `docs` branch if it doesn't already exist.

### 2. Clean the `docs` Branch

- In the `docs` branch, delete all files except for `.gitignore`.
- The branch should now be nearly empty, ready for the documentation setup.

### 3. Installing Dependencies & Configuring Sphinx

You no longer need to manually create the `src` folder and populate `.rst` files yourself. The Python script described below will handle this step automatically for you.

**Install Dependencies:**

```bash
pip install sphinx sphinx-automodapi pydata-sphinx-theme
```
````

**Run `sphinx-quickstart`:**

```bash
sphinx-quickstart
```

If the command is not found:

```bash
python -m sphinx.cmd.quickstart
```

This generates:

- `conf.py` (Sphinx configuration file)
- `Makefile`

**Edit `conf.py`:**

- Add your code to `sys.path` so Sphinx can locate it:
  ```python
  import sys
  sys.path.insert(0, "release-code")
  ```
- Add required extensions:

  ```python
  extensions = ["sphinx_automodapi.automodapi", "sphinx.ext.githubpages"]
  ```

- Set the theme and options:
  ```python
  html_theme = "pydata_sphinx_theme"
  html_static_path = ["static"]
  html_theme_options = {
      "switcher": {
          "json_url": "https://yourusername.github.io/yourproject/latest/_static/switcher.json",
          "version_match": version,
      },
      "navbar_end": [
          "version-switcher",
          "navbar-icon-links",
      ],
  }
  ```

**Create `index.rst`:**
Place an `index.rst` file at the root of the `docs` branch:

```rst
.. yourproject documentation master file.

yourproject documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   src/Documentation
```

## Generating `.rst` Files with the Script

A Python script `generate_doc_markdown.py` is provided to automate the `.rst` creation. It:

- Identifies top-level modules and packages in `release-code/<library_name>`.
- Generates `.rst` files for each module.
- Creates a `Documentation.rst` that acts as a table of contents.

**Usage:**

```bash
python generate_doc_markdown.py <library_name> [--force] [--exclude=mod1,mod2,...] [--output-dir=src]
```

**Arguments:**

- `<library_name>`: The name of your library (the folder under `release-code`).
- `--force` / `-f`: Overwrite existing `.rst` files if present.
- `--exclude` / `-e`: Exclude specific submodules from documentation.
- `--output-dir` / `-o`: Directory to write output files, default is `src`.

**Example:**

```bash
python generate_doc_markdown.py deeplay --force --exclude=_internal --output-dir=src
```

This command creates the `src` folder (if needed), generates `.rst` files for each module, and updates `Documentation.rst` automatically.

## The Version Switcher

The version switcher allows users to navigate between different versions of your documentation directly from the site’s navigation bar. This is handled by the switcher.json file, stored in static/switcher.json, which follows a structure like:

```
[
  {
    "name": "0.1.0",
    "version": "0.1.0",
    "url": "https://yourusername.github.io/yourproject/0.1.0/"
  },
]
```

How it works:

    Each entry in switcher.json specifies a documentation version and the URL where it can be found.
    The html_theme_options in conf.py references this file, enabling a dropdown menu to choose the version.
    The GitHub Actions workflow updates switcher.json upon new releases by prepending the new version into this list. This ensures that the newly released version appears in the version switcher, and users can easily switch to it.

## Automated Documentation on Release

Once your documentation setup is complete (i.e., you have `conf.py`, `index.rst`, `requirements.txt`, etc. in place), the provided GitHub Actions workflow automates the entire process whenever a new release is published. This workflow:

1. Checks out the `docs` branch.
2. Sets up Python and installs dependencies from `requirements.txt`.
3. Checks out the release code (tag specified by the release event) into `release-code`.
4. Runs the `generate_doc_markdown.py` script to generate/update `.rst` files.
5. Builds the Sphinx documentation.
6. Copies the built HTML files to `docs/latest` and `docs/{version}` directories.
7. Commits and pushes these changes back to the `docs` branch, updating the live documentation on GitHub Pages.

**Example Workflow (in `.github/workflows/update-docs.yml`):**

```yaml
name: Update Documentation

on:
  release:
    types:
      - published
  workflow_dispatch:
    inputs:
      test_tag:
        description: "Release tag to simulate"
        required: true

jobs:
  update-docs:
    name: Update Documentation
    runs-on: ubuntu-latest

    steps:
      # Step 1: Check out the docs branch
      - name: Checkout docs branch
        uses: actions/checkout@v3
        with:
          ref: docs

      # Step 2: Set up Python
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      # Step 3: Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r doc_requirements.txt

      # Step 4: Pull the release code into a separate directory
      - name: Checkout release code
        uses: actions/checkout@v3
        with:
          path: release-code
          # Use the test tag from workflow_dispatch or the actual release tag
          ref: ${{ github.event.inputs.test_tag || github.event.release.tag_name }}

      - name: Install the package
        run: |
          cd release-code
          pip install -e .

      - name: Create the markdown files
        run: |
          python generate_doc_markdown.py deeplay --exclude=tests,test

      # Step 5: Set version variable
      - name: Set version variable
        run: |
          VERSION=${{ github.event.inputs.test_tag || github.event.release.tag_name }}
          echo "VERSION=$VERSION" >> $GITHUB_ENV

      # Step 6: Update switcher.json
      - name: Update switcher.json
        run: |
          SWITCHER_FILE=static/switcher.json
          jq --arg version "$VERSION" \
             '. |= [{"name": $version, "version": $version, "url": "https://deeptrackai.github.io/deeplay/\($version)/"}] + .' \
             $SWITCHER_FILE > temp.json && mv temp.json $SWITCHER_FILE

      # Step 7: Build documentation using Sphinx into html
      - name: Build documentation
        env:
          SPHINX_APIDOC_DIR: release-code
        run: make html

      # Step 8: Copy built HTML to `docs/latest` and `docs/{version}`
      - name: Copy built HTML
        run: |
          mkdir -p docs/latest
          mkdir -p docs/$VERSION
          cp -r _build/html/* docs/latest/
          cp -r _build/html/* docs/$VERSION/

      # Step 9: Clean up `release-code` directory
      - name: Remove release-code directory
        run: rm -rf release-code

      # Step 10: Commit and push changes
      - name: Commit and push changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/latest docs/$VERSION static/switcher.json
          git commit -m "Update docs for release $VERSION"
          git push
```

## Logical Flow of Documentation Updates

1. **Initial Setup**: Configure `docs` branch, Sphinx, and dependencies.
2. **Automated `.rst` Generation**: Use `generate_doc_markdown.py` to generate `.rst` files from code.
3. **Building Locally (if needed)**: `make html` to verify documentation locally.
4. **Release Trigger**: On publishing a new release, the GitHub Actions workflow:
   - Checks out `docs` branch and release code.
   - Generates `.rst` files.
   - Builds HTML docs.
   - Deploys the updated docs to `docs/latest` and `docs/<version>`.
5. **Automatic Deployment**: Changes are committed back to `docs` branch and served on GitHub Pages.

## Special Files

- **`docs` branch**: Where documentation content and builds are hosted.
- **`index.rst`**: Root documentation file linking to `Documentation.rst`.
- **`Documentation.rst`**: Table of contents for your modules, generated automatically.
- **`conf.py`**: Sphinx configuration file where you set up paths, extensions, and themes.
- **`generate_doc_markdown.py`**: Automation script that eliminates the need for manual `.rst` creation.
- **`doc_requirements.txt`**: Lists the dependencies (Sphinx, sphinx-automodapi, pydata-sphinx-theme, etc.) needed to build documentation.
- **`Makefile`**: Provides convenient commands (`make html`) to build the docs locally.
- **`static/switcher.json`**: Used for version switching within the docs (managed by the workflow).
