with import <nixpkgs> {};

let
  cnpy = (pkgs.callPackage ./nix/cnpy.nix { });
in
stdenv.mkDerivation {
  name = "build-env";
  buildInputs = [
    cnpy
    zlib
  ];
}
