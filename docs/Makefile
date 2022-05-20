#
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXDOCSTR  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
APIDIR        = api
PACKAGEDIR    = ../sad

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

build:
	@$(SPHINXBUILD) -M build "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rmapi:
	rm -r ./source/api

html: clean
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@$(SPHINXDOCSTR) -o "$(SOURCEDIR)/$(APIDIR)" "$(PACKAGEDIR)" 

.PHONY: help build clean rmapi html apidoc

