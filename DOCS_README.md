# Docs how-to

## How to setup docs on a new repo

### 1. create the docs branch

Step one is to create the docs branch through the github interface.
This is done in the settings, under github pages. Set up so that the pages are hosted from the docs folder in the docs branch.

### 2. Delete everything in the docs branch

Delete everything in the docs branch, except for the .gitignore file. Nothing else is needed for now.

### 3. Create the src folder and populate it

Create the src folder in the root of the docs branch. This is where the markdown files will be stored.
For every python file and folder one level deep in the project root, create a markdown file with the same name in the src folder (with the .rst extension).
For models.py, the markdown file should be named models.rst and include the following:

```
.. automodapi:: deeplay.module
```

### 5. Create Documentation.rst

Create a Documentation.rst file in the src folder. This file should include the following:

```
Documentation
=============

Here, you will find the documentation for the Deeplay library.
The documentation is organized into the following sections:

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   models
   [all other files, in order]
```

### 6. Create index.rst

This will be in the root of the repo, should include the following:

```
.. deeplay documentation master file, created by
   sphinx-quickstart on Mon Aug 26 12:06:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

deeplay documentation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   src/Documentation

```

### 7. Install sphinx and the dependencies

We need: Sphinx, sphinx-automodapi, pydata-sphinx-theme
They do the following:

- Sphinx: the main package
- sphinx-automodapi: to automatically generate documentation for the modules
- pydata-sphinx-theme: a theme for the documentation

To install them, run the following commands:

```
pip install sphinx sphinx-automodapi pydata-sphinx-theme
```

### 8. Run sphinx-quickstart

Run the following command in the root of the docs branch:

```
sphinx-quickstart
```

if the script is not found, you can run it with:

```
python -m sphinx.cmd.quickstart
```

This will create a conf.py file and a Makefile. The conf.py file should be edited to include the following:

So it can find the code:

```
import sys

sys.path.insert(0, "release-code")
```

For extensions, add the following:

```
extensions = ["sphinx_automodapi.automodapi", "sphinx.ext.githubpages"]
```

at the bottom add

```
html_theme = "pydata_sphinx_theme"
html_static_path = ["static"]
html_theme_options = {
    "switcher": {
        "json_url": "https://deeptrackai.github.io/deeplay/latest/_static/switcher.json",
        "version_match": "latest",
    },
    "navbar_end": [
        "version-switcher",
        "navbar-icon-links",
    ],
}
```
