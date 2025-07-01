from panda3d.core import loadPrcFileData, WindowProperties
from direct.showbase.ShowBase import ShowBase

# /// script
# dependencies = [
#  "panda3d",
# ]
# ///

loadPrcFileData("", "load-display pandagl")
loadPrcFileData("", "gl-check-errors t")

base = ShowBase()
base.win.set_clear_color((0.2, 0.2, 0.6, 1))
base.win.request_properties(WindowProperties(foreground=True))
model = base.loader.loadModel("models/environment")
model.reparentTo(base.render)
model.setPos(0, 42, 0)
base.cam.setPos(0, -60, 10)
base.cam.lookAt(model)
base.run()