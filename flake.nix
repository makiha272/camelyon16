{
  description = "My first reproducible Python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05"; # パッケージセット
  };

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = nixpkgs.mkShell {
      buildInputs = [
        nixpkgs.python311
        nixpkgs.python311Packages.numpy
        nixpkgs.python311Packages.pandas
      ];
    };
  };
}