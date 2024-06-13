{ sources ? import ./sources.nix }:

import sources.nixpkgs {
  overlays = [
    (_: pkgs: { inherit sources; })
    (_: pkgs: {
      ocamlPackages = pkgs.ocamlPackages.overrideScope' (self: super: {
        menhirLib = self.callPackage ./menhir/lib.nix { };
        # menhirSdk = self.callPackage ./menhir/sdk.nix { };
        menhir = self.callPackage ./menhir/default.nix { };
      });
    })
  ];
}
