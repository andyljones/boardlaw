# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
print(sys.path)


# -- Project information -----------------------------------------------------

project = 'boardlaw'
copyright = '2020, Andy L. Jones'
author = 'Andy L. Jones'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'sphinx.ext.extlinks'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

html_theme_options = {
    'description': 'board game scaling laws',
    'fixed_sidebar': True,
    'github_button': True,
    'github_user': 'andyljones',
    'github_repo': 'boardlaw',
    'github_type': 'star',
    'github_count': False,
}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_style = 'custom.css'

autoclass_content = 'both'
autodoc_member_order = 'bysource'

extlinks = {'github': ('https://github.com/andyljones/boardlaw/tree/master/%s', '')}

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None

    filename = info['module'].replace('.', '/') + '.py'

    return f"https://github.com/andyljones/megastep/tree/master/{filename}"