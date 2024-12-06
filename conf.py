# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "deeplay"
copyright = "2024, Benjamin Midtvedt, Jesus Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Carlo Manzo, Giovanni Volpe"
author = "Benjamin Midtvedt, Jesus Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Carlo Manzo, Giovanni Volpe"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_automodapi.automodapi", "sphinx.ext.githubpages"]
numpydoc_show_class_members = False
automodapi_inheritance_diagram = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["static"]
html_theme_options = {
    "switcher": {
        "json_url": "https://deeptrackai.github.io/deeplay/latest/_static/switcher.json",
        "version_match": "0.1.1",
    },
    "navbar_end": [
        "version-switcher",
        "navbar-icon-links",
    ],
}
