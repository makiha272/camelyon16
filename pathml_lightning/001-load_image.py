# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def _():
    from pathml.core import HESlide
    wsi = HESlide("/workspace/dataset/camelyon16/images/normal_001.tif", backend="openslide")
    return (wsi,)


@app.cell
def _(wsi):
    import matplotlib.pyplot as plt
    wsi.plot()
    plt.show()
    return


@app.cell
def _():
    from pathml.preprocessing import Pipeline, BoxBlur, TissueDetectionHE

    pipeline = Pipeline(
        [
            BoxBlur(kernel_size=15),
            TissueDetectionHE(
                mask_name="tissue",
                min_region_size=500,
                threshold=30,
                outer_contours_only=True,
            ),
        ]
    )
    return (pipeline,)


@app.cell
def _(pipeline, wsi):
    wsi.run(pipeline, write_dir="/workspace/pathml_lightning/", distributed=False, tile_size=224)
    return


@app.cell
def _(wsi):
    help(wsi.run)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
