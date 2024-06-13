{ pkgs ? import ./nix { } }:

let
  inherit (pkgs) ocamlPackages;
in

pkgs.mkShell {
  strictDeps = true;

  configureFlags = [ "--enable-middle-end-flambda2" ];

  nativeBuildInputs = with ocamlPackages; [
    ocaml dune_3 menhir pkgs.which pkgs.rsync pkgs.autoconf pkgs.ocamlformat_0_24_1
  ];

  buildInputs = with ocamlPackages; [ menhirLib ];
}
