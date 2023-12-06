{ pkgs ? import <nixpkgs> { }, pythonPackages ? pkgs.python3Packages }:

let
  pythonChess = pythonPackages.buildPythonPackage rec {
    pname = "chess";
    version = "1.10";
    format = "wheel";
    src = pythonPackages.fetchPypi {
      inherit pname version;
      sha256 = "15cfcb738d2518daf04d34b23419bd359cbd8e09da50778ebac521774fc8";
       format = "wheel";
      python = "py3";
        abi = "none";
        platform = "any";
    };
  };
in
pkgs.mkShell {
  buildInputs = [
    pythonPackages.pip
    pythonPackages.numpy
    pythonPackages.scipy
    pythonPackages.jupyterlab
    pkgs.stockfish
    pythonChess
  ];
}
