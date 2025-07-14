from panda3d.core import loadPrcFileData, WindowProperties, Point3, Vec3
from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from direct.task import Task

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
model.setScale(0.25, 0.25, 0.25)
model.setPos(-8, 42, 0)
pandaActor = Actor("models/panda-model",
                   {"walk": "models/panda-walk4"})
pandaActor.setScale(0.005, 0.005, 0.005)
pandaActor.reparentTo(base.render)
pandaActor.loop("walk")
posInterval1 = pandaActor.posInterval(13,
                                      Point3(0, -10, 0),
                                      startPos=Point3(0, 10, 0))
posInterval2 = pandaActor.posInterval(13,
                                      Point3(0, 10, 0),
                                      startPos=Point3(0, -10, 0))
hprInterval1 = pandaActor.hprInterval(3,
                                      Point3(180, 0, 0),
                                      startHpr=Point3(0, 0, 0))
hprInterval2 = pandaActor.hprInterval(3,
                                      Point3(0, 0, 0),
                                      startHpr=Point3(180, 0, 0))
pandaPace = Sequence(posInterval1, hprInterval1,
                     posInterval2, hprInterval2,
                     name="pandaPace")
pandaPace.loop()
base.cam.setPos(0, -60, 10)
base.cam.lookAt(model)
base.run()