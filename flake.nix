{
  description = "A project, nothing more.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    alejandra = {
      url = "github:kamadorueda/alejandra/4.0.0";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    alejandra,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        src = pkgs.lib.cleanSource ./.;

        package = pkgs.rustPlatform.buildRustPackage {
          pname = "seaweed";
          version = "0.1.0";

          inherit src;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };
        };
      in {
        # nix build .
        packages.default = package;

        # nix run .#format
        apps.format = flake-utils.lib.mkApp {
          drv = pkgs.writeShellApplication {
            name = "format";

            runtimeInputs = [
              alejandra.defaultPackage.${system}
              pkgs.cargo

              pkgs.rustfmt
            ];

            text = ''
              alejandra ./flake.nix
              cargo fmt
            '';
          };
        };

        checks.build = package;
        checks.nix-format =
          pkgs.runCommand "nix-format" {
            inherit src;

            nativeBuildInputs = [
              alejandra.defaultPackage.${system}
            ];
          } ''
            mkdir -p $out
            alejandra --check ${./flake.nix}
          '';
        checks.rustfmt =
          pkgs.runCommand "rustfmt" {
            inherit src;

            nativeBuildInputs = [
              pkgs.cargo
              pkgs.rustfmt
            ];
          } ''
            mkdir -p $out
            cd ${src}
            cargo fmt --check
          '';
      }
    );
}
