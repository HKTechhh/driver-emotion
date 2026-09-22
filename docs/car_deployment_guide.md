<!--
DRAFT STATUS (delete this block before sharing)
- This guide is engineering advice written from the project's own results (see paper/paper.md), not a
  certified automotive design. Prices are rough bands, not quotes, and will drift - check current listings.
- Anything that involves cutting into a vehicle's wiring, fuse box or airbag zone should be reviewed by
  someone qualified to work on that specific car, or done through a professional installer.
-->

# From Laptop Demo to In-Car Prototype: A Deployment Guide

This is a practical guide to taking the driver-emotion model out of the lab and into an actual vehicle,
written for whoever picks up this project next. It assumes you have `cnn_v1.keras` (or `vgg16_v1.keras`)
and a working `src/realtime.py`, and it walks through the hardware, wiring, mounting and software choices,
plus the safety and legal issues that come with pointing a camera at a driver. It is not a certified
automotive design, and none of what follows should be read as a claim that this system is ready to
influence how someone actually drives.

## 1. Prove it cheaply before you build anything

Before buying hardware, mount a laptop or an old phone on the dashboard with a suction mount, run
`src/realtime.py` off USB battery power, and drive around (as a passenger, not the driver) with someone
else at the wheel. This answers the questions that matter before you spend money: does the camera angle
actually see the driver's face at different seat heights, does the console glare in mid-afternoon sun wash
out the feed, does the fan noise or heat become a problem on a warm day, and does the model's behaviour
on `data/raw/test` images match what you see on a real face in a moving car. Only move on to a dedicated
embedded box once this cheap version behaves the way you expect.

## 2. Choosing the compute

The custom CNN is small (4.8 M parameters) and, as measured in Section 4.4 of the paper, runs at about
25 FPS for the model alone on a 2019 laptop CPU (Intel Core i5-8265U) with no GPU. That is comfortably
within reach of a single-board computer, so a full in-car PC is overkill for this model.

| Option | Rough cost | Notes |
|---|---|---|
| Raspberry Pi 5 (8 GB) + active cooler | $80-100 | The default recommendation. Runs Linux, has a camera connector (CSI) and USB ports, and is well documented. Expect noticeably fewer FPS than the laptop figure above - budget time to re-measure and possibly convert the model to TensorFlow Lite and drop the input resolution path for VGG16 if you want to run that model instead of the CNN. |
| NVIDIA Jetson Orin Nano (Super) | $250 or more | Overkill for this specific CNN, but worth it if you plan to add a second model (drowsiness, gaze, lane-departure) that needs a real GPU. |
| Old Android phone, tethered or dedicated | $0 if you already own one | Cheapest option and good for the cheap-first-test in Section 1. Long-term reliability (heat, battery degradation, OS updates changing camera APIs) makes it a worse choice for something meant to run every drive for months. |

Whichever board you pick, plan for heat: an enclosed dashboard in direct sun in summer easily exceeds
60°C, well above what a bare Raspberry Pi is comfortable running at continuously. Get a case with a fan or
a heatsink, and do not seal it inside an airtight box.

## 3. Camera

The paper's robustness results (Section 4.5, Table 7) are the single most important piece of evidence
for this decision: low light cut both models to roughly the accuracy of always guessing the majority
class, and that was already the least confident condition tested. A car cabin at night, lit only by
dashboard glow and passing headlights, is worse than the simulated `low_light` condition. A plain visible-
light webcam will not work after dark.

The usual fix in commercial driver-monitoring systems is a near-infrared (NIR) camera with its own IR LED
illuminator, which works in complete darkness and is not blinding to the driver. This is a genuine
hardware fix, but it comes with a catch worth stating plainly: **the model in this project was trained
entirely on visible-light FER2013 images.** An IR camera produces a different-looking image (no colour,
different contrast, specular reflections off glasses), and this project has not tested, let alone
retrained, on IR footage. Buying an IR camera does not by itself fix the low-light result - it changes the
problem to "train or fine-tune on IR data," which is listed as future work in the paper (Section 5.5) and
is a project of its own, not a weekend task.

| Option | Rough cost | Notes |
|---|---|---|
| Generic USB webcam (720p/1080p) | $15-30 | Fine for the daytime bench test in Section 1. Reproduces the "clean" condition in the paper, not the low-light one. |
| Raspberry Pi NoIR Camera Module 3 + IR illuminator board | $35-60 combined | CSI-connected, no IR-cut filter, so it can see under IR light. Needed for a night-capable prototype, with the retraining caveat above. |
| Dedicated automotive driver-monitoring camera module | $50-150+ | Sold for fleet dashcams; usually includes IR LEDs and a wide-angle lens sized for a dashboard mount. Overkill unless you are building something you intend to leave installed long-term. |

## 4. Power

Do not run the board off its own battery for anything beyond a bench test; a Raspberry Pi under load for
a whole commute will outlast a phone power bank but not comfortably. Two realistic options, in order of
how invasive they are:

1. **Cigarette-lighter / 12 V accessory socket USB adapter.** Plug-and-play, no tools, easy to remove.
   Get one rated for at least 3 A at 5 V (Raspberry Pi 5 wants a 5V/5A supply under full load, so budget
   for a higher-rated adapter than a phone charger). This is the right choice for a prototype and for
   anyone who does not want to open up the car's dashboard.
2. **Hardwired to a switched (ignition/ACC) 12 V circuit**, stepped down to 5 V with a buck converter, so
   the board powers on and off with the car rather than needing to be plugged in and unplugged. This is
   what a permanent installation looks like, and it is also the step where you can do real damage: pulling
   power from the wrong fuse, wiring into an airbag or safety-restraint circuit, or shorting something
   near the steering column, is a genuine risk to the car and to you. If you want a permanent install,
   use a proper add-a-fuse tap kit rated for the car's fuse box, confirm with a multimeter that the fuse
   you tap is switched (dead with the ignition off, live with it on) before connecting anything, and if
   you are not confident doing that yourself, pay an auto electrician or car-audio installer to do this
   one step. A cheap OBD-II power-tap adapter (plugs into the OBD-II diagnostic port most cars have under
   the dash) is a lower-risk middle ground that gives switched 12 V without touching the fuse box directly.

Either way, add an inline fuse close to the power source, and if the board writes to an SD card, plan for
what happens if power cuts out mid-write when the driver turns the engine off - repeated sudden power loss
corrupts SD cards. A small UPS hat with a supercapacitor (several exist for the Raspberry Pi) that keeps
the board alive for a clean shutdown after ignition-off is worth the extra $20-40 for anything left
installed longer than a few days.

## 5. Mounting the camera

Aim for the same framing the model was trained toward: a roughly frontal view of the driver's face. In
practice that means near the top of the dashboard, on the A-pillar, or built into the rear-view mirror
housing, angled down slightly at the driver's eye level. Two hard constraints:

- **Never mount anything, or route any cable, through the airbag's deployment zone** (the steering wheel
  hub, the top of the dashboard directly ahead of the passenger, the A-pillar if the car has side-curtain
  airbags routed through it). Check the car's manual for exactly where those zones are for that model.
- Keep the camera and its cable out of the driver's normal sightline and out of the path of the wipers'
  view through the windscreen if it is mounted near the top of the glass.

A dashcam-style adhesive or suction mount for the camera module, with the cable run along the A-pillar
trim and under the dashboard edge (both can usually be gently pried back and reattached without tools),
gives a tidy result without permanent modification. If the board itself needs to be hidden, the glovebox
or under-dash area works, as long as the camera cable is long enough to reach the mounting point (CSI
cables have length limits before signal quality drops - a USB camera has more flexibility here).

## 6. Software on the board

1. Flash a 64-bit Raspberry Pi OS (or the Jetson equivalent) image, enable the camera interface, and
   confirm `raspistill`/`libcamera-hello` (or the Jetson camera tool) can see a frame before installing
   anything from this project.
2. Install the same Python dependencies as `requirements.txt`, noting that TensorFlow's ARM wheels are
   slower to install than on x86 and that MediaPipe's prebuilt wheels do not cover every board revision -
   check MediaPipe's own install instructions for the board you bought.
3. For real headroom on a Raspberry Pi, convert `cnn_v1.keras` to TensorFlow Lite
   (`tf.lite.TFLiteConverter.from_keras_model`) and, if the frame rate still is not enough, try dynamic-
   range or full-integer post-training quantization. Re-measure accuracy on the test set after conversion;
   quantization can cost a point or two of accuracy, and the paper's Table 4/5 numbers do not apply to a
   quantized model without re-checking.
4. Wrap `src/realtime.py`'s camera loop in a `systemd` service (`Restart=on-failure`) so it starts
   automatically at boot and restarts itself if the camera briefly disconnects, rather than needing
   someone to SSH in and start it manually every drive.

## 7. Making the alert safe, not just present

The current `--alert-seconds 3.0` behaviour draws a full-screen red banner over the video feed
(`config.py`'s `REALTIME` settings). That is fine for a laptop demo where nobody is meant to look at the
screen while driving, but it is close to the opposite of what you want in a car: a bright, sudden visual
element in a driver's peripheral vision is a distraction, not a safety feature, especially if the screen
sits anywhere near the driver's forward view. Two changes are worth making before this goes anywhere near
an actual dashboard:

- Replace or supplement the visual banner with an audio cue - a short chime through a small speaker, or
  routed over the car's own Bluetooth or aux input if it has one - so the driver does not need to look at
  anything to notice it.
- Do not mount a screen showing the live camera feed, probability bars or bounding box anywhere the driver
  can see it while driving. Log that information to CSV (as the app already does) for later review instead
  of displaying it live; a live emotion readout is a second thing competing for the driver's attention,
  which undermines the entire point of the system.

## 8. Legal and privacy considerations

This section is general orientation, not legal advice - check the rules that apply where the car is
registered and used before installing anything.

- **Recording someone's face is processing biometric-adjacent data** in many jurisdictions, which can carry
  stricter consent and storage rules than an ordinary dashcam. If the driver is not you (a shared car, a
  fleet vehicle, a rental, a family member's car), get their informed consent before switching this on,
  and tell them plainly what is stored (per Section 6 of the paper: no images are saved, only a timestamp,
  the predicted emotion, its confidence and the frame rate).
- Some regions restrict or require disclosure for any camera pointed at a vehicle's occupants, separate
  from data-protection law. Check local vehicle and traffic regulations, not just privacy law.
- If this is ever used in a work vehicle, workplace-monitoring rules (and, in many places, a legal
  requirement to consult the employee or their representatives before deploying it) apply on top of
  general privacy law.
- Treat the model's output as a weak, probabilistic signal, not a fact about the driver's state (this
  point is also made in Section 6 of the paper), and do not build anything that penalises a driver -
  insurance premiums, employment decisions, automatic vehicle lockouts - directly from its output.

## 9. Testing before you trust it

1. **Bench test, engine off.** Confirm the camera sees the driver's seat properly, the board boots into
   the service on its own, and the alert (visual or audio) fires when you deliberately hold an angry or
   fearful expression for the configured 3 seconds.
2. **Parked, engine on.** Check that vibration from the idling engine does not loosen the mount, and that
   the board's power draw does not trip anything when the car's other accessories (AC, infotainment) are
   also running.
3. **Supervised, low speed, with a passenger watching the log rather than the road.** Never test this
   alone while you are the one driving - have someone else drive while you watch the laptop, or watch the
   log afterwards, so nobody's attention is divided by a system that has not been validated yet.
4. Only after that should the camera and alert run during an ordinary drive, and even then, treat every
   output as something to sanity-check against how the drive actually went, not as ground truth.

## 10. Bill of materials (indicative)

| Tier | Contents | Rough total |
|---|---|---|
| **Prototype** (Section 1) | Existing laptop or old phone, suction mount, USB power bank | $0-20 |
| **Daytime embedded box** | Raspberry Pi 5 + case/cooler, USB webcam, cigarette-lighter USB power adapter, SD card, dashcam suction mount | $130-170 |
| **Night-capable, semi-permanent install** | Raspberry Pi 5 + case/cooler, NoIR camera + IR illuminator, OBD-II or fuse-tap 12V power with buck converter, small UPS hat, small speaker, mounting hardware and cable ties | $250-350, plus IR retraining work not covered by any of this hardware |

Prices move; treat these as planning bands from when this was written, not quotes, and re-check current
listings before buying.

## 11. What this does not solve

None of the hardware above closes the gaps the paper is explicit about (Section 5.5): the model was
trained and tested on FER2013 web photos, not in-cabin footage; the driving conditions in Section 4.5 are
simulated corruptions of those photos, not real recordings from a moving car; each model was trained once,
so its behaviour on a different day's lighting or a different face shape is untested; and the one live
session run so far (Section 3.9, Appendix in the experiment log) was 42 minutes with one person in one
room, not a validated in-vehicle evaluation. Building the box in this guide gets the model physically
running in a car. It does not, by itself, make the model's predictions trustworthy enough to act on.
