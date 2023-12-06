{ pkgs ? import <nixpkgs> { }, pythonPackages ? pkgs.python3Packages }:

let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix";
    ref = "refs/tags/3.5.0";
  }) {};
in 
pkgs.mkShell {
  buildInputs = [
      pkgs.stockfish
      (mach-nix.mkPython {
        requirements = ''
          python-chess
          numpy
          jupyter
        '';
      })
  ];
}
