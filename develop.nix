/* develop.nix: Specifies a development environment.  To use it:

$ cd 'this dir'; nix-shell -j6 --pure --keep NIX_PATH develop.nix

See Language Server keys at:
https://emacs-lsp.github.io/lsp-mode/page/keybindings/
*/
with import <nixpkgs> {}; # bring all of Nixpkgs into scope

let
  # See https://nixos.wiki/wiki/Python, and note the similar
  # definition of emacs-with-packages.
  python-select = python-packages:
    with python-packages; [
      scipy numpy matplotlib pyqt5 ipython ipdb setuptools
      sphinx pytest cython pip pint pytestcov pylint yapf mypy ];

  python-with-my-packages =
    python37.withPackages python-select;

  EmacsConfig = import lib_nix/emacs_config.nix writeText;
  EmacsSelect = import lib_nix/emacs_select.nix {inherit runCommand EmacsConfig;} ;
  emacs-with-packages = (emacsPackagesNgGen emacs).emacsWithPackages EmacsSelect ;

  myaspell = aspellWithDicts (d: [d.en]);

   /* From https://nixos.org/nixpkgs/manual/#sec-pkgs-mkShell:
"pkgs.mkShell is a special kind of derivation that is only useful
when using it combined with nix-shell."  The suggested command,
"nix-shell -j6 --pure develop.nix" drops the user into a shell
with a "pure" environment that has the executables listed in
buildInputs in the PATH.  The -j6 options says to run up to 6
build jobs simultaneously. */

in pkgs.mkShell rec {
  buildInputs = [
    file
    ncurses5
    firefox
    git
    nix
    less
    texlive.combined.scheme-tetex
    gnumake42
    bash
    which
    evince
    myaspell
    python-with-my-packages
    emacs-with-packages
    less
    python-language-server
    man
    wget
  ];
  shellHook = ''
    export PYLINTRC=`realpath pylintrc`
    export PYTHONPATH=`realpath ./.`:$PYTHONPATH
  '' ;
}

/* I'd like to run python setup.py develop, but I don't know how to
get nix to do that as it also sets up a pure nix shell with selected
development tools.  I think that the setup.nix project at
https://github.com/nix-community/setup.nix does what I want, but it's
deprecated.  I don't want to invest in learning a deprecated tool.  So
I just hack PYTHONPATH.

*/
