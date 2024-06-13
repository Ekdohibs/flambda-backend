{ pkgs ? import ./nix { } }:

let
  inherit (pkgs) lib stdenv ocamlPackages fetchFromGitHub;
  pname = "ocaml+flambda2";
  version = "5.1.1+jst";
  src = fetchFromGitHub {
    owner = "ocaml-flambda";
    repo = "flambda-backend";
    rev = "771f0bc0ad8ca6a8c602a26d96d4b0806f2f7329";
    hash = "sha256-3Hm4rpMoIBhgvq3s/kyp9GV+i9aunvIBreWsyOmU794=";
  };
in

stdenv.mkDerivation {
  inherit pname version src;

  strictDeps = true;

  configureFlags = [ "--enable-middle-end-flambda2" ];

  nativeBuildInputs = with ocamlPackages; [
    ocaml dune_3 menhir pkgs.which pkgs.rsync pkgs.autoreconfHook
  ];

  buildInputs = with ocamlPackages; [ menhirLib ];
}
