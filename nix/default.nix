import (builtins.fetchTarball {
  url = "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.05.tar.gz";
  sha256 = "1lr1h35prqkd1mkmzriwlpvxcb34kmhc9dnr48gkm8hh089hifmx";
}) {
  overlays = [
    (_: pkgs: {
      ocamlPackages = pkgs.ocamlPackages.overrideScope' (self: super: {
        menhirLib = self.callPackage ./menhir/lib.nix { };
        # menhirSdk = self.callPackage ./menhir/sdk.nix { };
        menhir = self.callPackage ./menhir/default.nix { };
      });
    })
  ];
}
