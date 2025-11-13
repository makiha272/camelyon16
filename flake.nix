{
  description = "Nix + uv hybrid Python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs, ... }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      # Nix が提供する「地盤レイヤー」
      buildInputs = with pkgs; [
        python311              # Python本体
        uv                     # Pythonパッケージマネージャ
        pkg-config             # C拡張ビルド用
        gcc                    # 一部のPythonパッケージが必要
        openssl
        zlib
      ];

      # シェル起動時に少しだけメッセージ
      shellHook = ''
        echo "🚀 Nix + uv hybrid environment ready!"
        echo "Python: $(python --version)"
        echo "uv: $(uv --version)"
      '';
    };
  };
}