"""Draw UI text at the window's real resolution, independently of pixel sprites."""

from PIL import Image, ImageDraw
from .art import font


class Surface:
    def __init__(self, size, color, scale=1.0):
        self.size = size
        self.scale = scale
        self.image = Image.new(
            "RGBA",
            (max(1, round(size[0] * scale)), max(1, round(size[1] * scale))),
            color,
        )

    def alpha_composite(self, other, dest=(0, 0), size=None, smooth=False):
        if isinstance(other, Surface):
            pixels = other.image
        else:
            logical = size or other.size
            target = (
                max(1, round(logical[0] * self.scale)),
                max(1, round(logical[1] * self.scale)),
            )
            pixels = other.resize(
                target, Image.Resampling.LANCZOS if smooth else Image.Resampling.NEAREST
            )
        self.image.alpha_composite(
            pixels, (round(dest[0] * self.scale), round(dest[1] * self.scale))
        )


class Draw:
    def __init__(self, surface):
        self.scale = surface.scale
        self.canvas = ImageDraw.Draw(surface.image)

    def coords(self, xy):
        if isinstance(xy[0], (tuple, list)):
            return [(round(x * self.scale), round(y * self.scale)) for x, y in xy]
        return tuple((round(v * self.scale) for v in xy))

    def options(self, kwargs):
        return {
            k: max(1, round(v * self.scale)) if k in ("width", "radius") else v
            for k, v in kwargs.items()
        }

    def rectangle(self, xy, **kwargs):
        self.canvas.rectangle(self.coords(xy), **self.options(kwargs))

    def rounded_rectangle(self, xy, **kwargs):
        self.canvas.rounded_rectangle(self.coords(xy), **self.options(kwargs))

    def ellipse(self, xy, **kwargs):
        self.canvas.ellipse(self.coords(xy), **self.options(kwargs))

    def pieslice(self, xy, start, end, **kwargs):
        self.canvas.pieslice(self.coords(xy), start, end, **self.options(kwargs))

    def line(self, xy, **kwargs):
        self.canvas.line(self.coords(xy), **self.options(kwargs))

    def polygon(self, xy, **kwargs):
        self.canvas.polygon(self.coords(xy), **self.options(kwargs))

    def draw_text(self, xy, value, size, fill, bold=False, anchor=None):
        self.canvas.text(
            self.coords(xy),
            value,
            font=font(max(1, round(size * self.scale)), bold),
            fill=fill,
            anchor=anchor,
        )


def draw(surface):
    return Draw(surface)
