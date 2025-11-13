import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import openslide
    from PIL import Image
    return (openslide,)


@app.cell
def _(openslide):
    slide = openslide.OpenSlide("/workspace/data/images/tumor_016.tif")
    print(slide.dimensions)  # (width, height)
    print(slide.level_count) # 解像度レベル数
    return (slide,)


@app.cell
def _(slide):
    for level in range(slide.level_count):
        print(f"Level {level}: size={slide.level_dimensions[level]}, downsample={slide.level_downsamples[level]}")
    return


@app.cell
def _(slide):
    thumbnail = slide.get_thumbnail((1024, 1024))
    thumbnail.show()
    return


@app.cell
def _(
    data,
    deepzoom,
    deepzoom_server,
    examples,
    images,
    openslide,
    python,
    root,
    tumor_016,
    workspace,
):
    python /root/openslide-python/examples/deepzoom/deepzoom_server.py /workspace/data/images/tumor_016.tif
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
