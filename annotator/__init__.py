"""Model-based control image preprocessors ("annotators"), ported from the original qDiffusion server.

Every annotator class is built as Class(folder, download) -- it downloads its own weights the first time -- and used as
    model.to(device, dtype); image = model(numpy_rgb_image, *args)
Submodules are imported lazily, so a missing optional package only breaks the preprocessors that need it.
"""
import importlib

ANNOTATORS = {
    "Anyline": ("annotator.anyline", "AnylineDetector"),
    "Lineart": ("annotator.lineart", "LineartDetector"),
    "Lineart Anime": ("annotator.lineart_anime", "LineartAnimeDetector"),
    "Softedge": ("annotator.hed", "HEDdetector"),
    "Scribble": ("annotator.hed", "HEDdetectorScribble"),
    "M-LSD": ("annotator.mlsd", "MLSDdetector"),
    "Depth": ("annotator.midas", "MidasDetector"),
    "Pose": ("annotator.openpose", "OpenposeDetector"),
}

def get_annotator_class(name):
    module, cls = ANNOTATORS[name]
    return getattr(importlib.import_module(module), cls)
