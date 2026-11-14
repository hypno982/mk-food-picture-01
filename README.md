# mk-food-picture — food-picture-01

One picture of food, drawn by a single Python 3.13 program with the standard library only:
a stack of three pancakes on a glazed plate, butter melting on top with its liquid halo and a
slumped corner, syrup pooled on the top disc and breaking over one edge as a thick pour with a
thinner thread beside it, blueberries (one halved, showing its flesh and seeds), and steam.
Every pixel is computed by hand and packed with `zlib`; there is no image library and no asset
file.

Three things make it read as food rather than as flat vector art: each pancake is a solid body
painted under the disc above it, a cast shadow falls from each disc onto the one below (and a
tight contact shadow grounds the stack on the plate), and a film grain weighted per material
sits over the whole dish.

## Run it

```sh
python3 main.py out.png
```

With no argument it writes `$MAKEFUN_OUTPUT_DIR/entry.png`, defaulting to `/output/entry.png`.

No network, no clock, no randomness: two runs write byte-identical files (731034 bytes,
md5 `e1b3af3d3970039dd1af23e0bef3931e`). 1024x1024, well under the 1 MiB gate, about 18 s.

`entry.png` in this repository is that render, committed so the picture is visible here.

## License

MIT, see `LICENSE`.
