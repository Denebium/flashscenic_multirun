"""Sphinx configuration for flashscenic documentation."""

project = "flashscenic"
copyright = "2025-2026, Hao Zhu, Donna Slonim"
author = "Hao Zhu, Donna Slonim"
version = "0.2.0"

extensions = [
    "myst_nb",
    "autodoc2",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Notebook settings: render pre-executed notebooks without re-running
nb_execution_mode = "off"

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# autodoc2 settings
autodoc2_packages = [
    "../flashscenic",
]
autodoc2_render_plugin = "myst"

# Theme
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/haozhu233/flashscenic",
    "use_repository_button": True,
    "use_issues_button": True,
}
html_title = "flashscenic"

# Source suffix — myst-nb auto-registers .md and .ipynb parsers
source_suffix = [".rst", ".md", ".ipynb"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
