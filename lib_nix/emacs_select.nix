# Return a function that maps a set of sets to a list of strings

/* The nix expression in this file evaluates to a function.  The
     argument of the function is a set of sets, eg,
     emacs-packages.melpaStablePackages.magit is a thing that gets the
     emacs package magit installed by a derivation.  The return value
     of the function is a list of things to install in a derivation
     for emacs.  See:
     https://nixos.org/nixos/manual/index.html#module-services-emacs
     */

{runCommand, EmacsConfig }:

# The argument of the function this file yields is called
# emacs-packages.  Its scope is local to this file.
emacs-packages:
(
  with emacs-packages.melpaStablePackages;
  [
    magit          # ; Integrate git <C-x g>
    nix-mode
    flycheck
    flycheck-pycheckers
    use-package
  ]
)
++ # Concatenate
(
  with emacs-packages.melpaPackages;
  [
    flycheck-pyflakes
  ]
)
++
(with emacs-packages.elpaPackages;
  [
    undo-tree      #  <C-x u> to show the undo tree
    #auctex         #  LaTeX mode
    beacon         # highlight my cursor when scrolling
  ]
)
++
[
  (runCommand "default.el" {} ''
mkdir -p $out/share/emacs/site-lisp
cp ${EmacsConfig} $out/share/emacs/site-lisp/default.el
'')
]
