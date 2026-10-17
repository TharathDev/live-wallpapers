# Ocean Run: live wallpaper

A relaxing, cinematic ocean scene for your laptop, with boats you can swap. The boat stays in place on screen and only rocks and shudders on the swell, while the sea streams past underneath it, cut into a white wake.

The whole animation is a **seamless loop** of 60 seconds by default, or 30 seconds with `?loop=30`. The last frame matches the first frame pixel for pixel, so it plays forever without a visible jump.

Inspired by [Noniv/web-ocean](https://github.com/Noniv/web-ocean). That project is a WebGPU FFT ocean that you fly around in. This one is a single WebGL2 file, so it also runs inside wallpaper apps and older browsers.

| Motor yacht, tropical, drone view | Galleon, golden hour, side view |
|---|---|
| ![](docs/yacht.jpg) | ![](docs/galleon.jpg) |
| **Fishing boat, overcast, deck view** | **Sailing yacht, clear day, chase view** |
| ![](docs/fishing.jpg) | ![](docs/sailboat.jpg) |

## Boats, scenes and cameras

Everything can be mixed. Each boat starts with the scene and camera that suit it best.

| Boat (`boat=`) | Speed | Default scene | Default camera |
|---|---|---|---|
| `yacht`: motor yacht with flybridge, twin prop wash, bow spray | 12 m/s | `tropical` | `drone` |
| `galleon`: three-masted square-rigger with braced yards, jibs and rigging | 3.5 m/s | `golden` | `side` |
| `fishing`: workboat with a rusty deck, rods over the wake, cooler, jerrycan | 7 m/s | `overcast` | `deck` |
| `sailboat`: heeled sloop with mainsail and genoa | 4.5 m/s | `clear` | `chase` |

- **Scenes** (`scene=`): `tropical`, `golden`, `overcast`, `clear`, `sunset`, `storm` (with lightning), `night` (moonlight, stars and nav lights).
- **Cameras** (`view=`):
  - `chase`: behind and above.
  - `drone`: straight down.
  - `side`: abeam.
  - `deck`: on board and rocking with the boat. For the galleon, whose deck is surrounded by sails, `deck` is a close-up from the waterline instead.

## Run it

Open `index.html` in a browser. It is self-contained and works from `file://` with no server and no install.

| Key | Action |
|---|---|
| `B` | Next boat (switches to its default scene and camera) |
| `S` | Next sky and sea |
| `C` | Next camera |
| `F` | Fullscreen |
| `R` | Record exactly one loop to a `.webm` video, starting at the loop seam |
| `H` | Show or hide the info overlay |

## Use it as your wallpaper

Point your wallpaper app at `index.html` with the options you like, for example `index.html?boat=galleon&scene=sunset`.

- **Windows:** [Lively Wallpaper](https://www.rocksdanister.com/lively/) (free). Choose *Add wallpaper*, then pick `index.html`, or paste the GitHub Pages URL.
- **macOS:** [Plash](https://sindresorhus.com/plash) (free). Add a website pointing at the file or the hosted URL.
- **Linux:** [Hidamari](https://github.com/jeffshee/hidamari) or Komorebi (web page mode). On KDE, use an HTML wallpaper plugin.
- **Video wallpaper apps:** press `R` and wait one loop. The saved `.webm` loops seamlessly in any player that repeats video, such as Wallpaper Engine, VLC, or Hidamari's video mode. For a crisp recording, open with `?scale=1` first.

## URL options

| Option | Default | Meaning |
|---|---|---|
| `boat` | `yacht` | `yacht`, `galleon`, `fishing` or `sailboat` |
| `scene` | depends on the boat | See the list of scenes above |
| `view` | depends on the boat | `chase`, `drone`, `side` or `deck` |
| `loop` | `60` | Loop length in seconds. Use 30 or 60, or anything from 10 to 300. |
| `speed` | depends on the boat | Boat speed in m/s, which sets how fast the sea streams past |
| `fps` | `60` | Frame-rate cap. Use `30` to save battery. |
| `scale` | auto | Render resolution relative to the screen. Leave unset for adaptive quality, or force it with values like `0.5` or `1`. |
| `fov` | depends on the camera | Vertical field of view in degrees |
| `exposure` | `1` | Brightness multiplier |
| `spray` | `1` | Bow spray multiplier. `0` turns it off. |
| `seed` | `7` | Different random sea |
| `grain` | `2` | Film grain strength. `0` turns it off. |
| `ui` | | `1` shows the overlay on start |
| `t`, `still` | | `t=12&still=1` renders one fixed frame, for screenshots |
| `cam`, `tgt` | | Camera position and look-at point, as `x,y,z` in metres (the boat heads +x) |

## How the seamless loop works

- **Waves:** the sea is a sum of 36 sharp-crested waves. Each wave's phase speed combines deep-water dispersion and the boat moving over it. That speed is rounded to a whole number of cycles per loop, so at `t = loop` every wave has returned to exactly where it started.
- **Boat motion:** heave, pitch and roll are sampled from those same waves under the hull, so the motion is periodic too. The deck camera rides on that motion. Engine vibration and the slow sway use whole-number frequencies per loop.
- **Spray:** each particle's lifetime divides the loop length exactly.
- **Foam and clouds:** whitecaps, wake foam, wake turbulence and clouds use noise that moves with the water. Two copies offset by one loop are cross-faded, which repeats with no seam.
- **Wake:** the bow wave, Kelvin wake arms and prop wash are fixed in the boat's frame, so they never break the loop.

## Performance

The scene is a full-screen ray-marched shader, and the boats are signed-distance models. The resolution adapts automatically to hold a steady frame rate, and rendering pauses when the page is hidden. The galleon, with all its sails, is the heaviest boat. On a laptop with integrated graphics, try `?fps=30` or `?scale=0.6`.
