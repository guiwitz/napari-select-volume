# napari-select-volume

A napari plugin to crop and rotate subvolumes in 3D images.

> [!WARNING]
> **This plugin is still in development and not intended for general use.** Use at your own risk.

## Installation

### pixi

You can use pixi to simplify the installation of the plugin and all extensions. For this:
1. Install pixi: https://pixi.prefix.dev/latest/installation/
2. Create folder and type `pixi init` to create a pixi.toml file.
3. To install the plugin, type `pixi add python=3.13 napari pyqt` then `pixi add --pypi "napari-select-volume @ git+https://github.com/guiwitz/napari-select-volume.git"`
4. Launch pixi with `pixi run napari` and open the plugin from the menu.

### conda
First create an environment for napari. For example with conda:

```bash
conda create -n napari-select-volume python=3.12 napari pyqt -c conda-forge
conda activate napari-select-volume
pip install git+https://github.com/guiwitz/napari-select-volume.git
```

## Author

This plugin was developed by [Guillaume Witz](https://github.com/guiwitz). Parts of the code were developed using Claude Sonnet 4.6. The whole code was reviewed and edited by Guillaume Witz.