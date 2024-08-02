{ stdenv, fetchFromGitHub, cmake, makeWrapper, lib, zlib }:

stdenv.mkDerivation rec {
  pname = "cnpy";
  version = "1.0";

  src = fetchFromGitHub {
    owner = "rogersce";
    repo = "cnpy";
    rev = "master";
    sha256 = "sha256-NMPDpeNoqvqAhwQk4J+TFw+BtNLI4R+CXpzXQ6hB/LU="; # Replace with actual SHA256 hash
  };

  nativeBuildInputs = [ cmake makeWrapper zlib ];

  cmakeFlags = [ "-DCMAKE_INSTALL_PREFIX=$out" ];

  buildPhase = ''
    cmake . -DCMAKE_INSTALL_PREFIX=$out
    make
  '';

  installPhase = ''
    make install
  '';

  meta = {
    description = "A C++ library for reading and writing NumPy .npy and .npz files";
    homepage = "https://github.com/rogersce/cnpy";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ exampleMaintainer ];
  };
}
