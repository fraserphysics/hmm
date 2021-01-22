/* emacs_confg.nix takes as its argument a function produce text and
 returns the value of that function applied to the text here, namely
 an emacs config file.

I set the local name of the argument to this function is "argument".
The file develop.nix imports/calls this file with the argument bing
the function "writeText".

I derived the following emacs configuration from the example in
https://nixos.org/nixpkgs/manual/
*/
argument:

argument "default.el" ''
;; initialize package

(require 'package)
(package-initialize 'noactivate)
(eval-when-compile
  (require 'use-package))

;; load some packages

(use-package flycheck
  :defer 2
  :config (global-flycheck-mode))

(defun init-flycheck ()
  (flycheck-mode t)
  (setq flycheck-pylintrc (getenv "PYLINTRC")
  )
  (flycheck-select-checker 'python-pylint)
  )
(add-hook 'python-mode-hook 'flycheck-mode)

(setq auto-mode-alist
(
 mapcar 'purecopy
  '(
    ("\\.tex$" . TeX-mode)
    ("\\.sh$" . shell-script-mode)
    ("\\.nix$" . nix-mode)
    ("\\.py\\'" . python-mode)
    ("Makefile" . makefile-mode)
    )
  )
)
;; from https://emacs-lsp.github.io/lsp-python-ms/
(use-package lsp-python-ms
  :ensure t
  :hook (python-mode . (lambda ()
                         (require 'lsp-python-ms)
                         (lsp)))
  :init
  (setq lsp-python-ms-executable (executable-find "python-language-server")))
  (setq w32-rwindow-modifier 'super)
''
