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
props = WindowProperties()
props.setCursorHidden(True)
# props.setMouseMode(WindowProperties.M_relative)  # Lock to window center
base.win.requestProperties(props)
key_map = {"w": False, "s": False, "a": False, "d": False}
for key in key_map:
    base.accept(key, lambda k=key: key_map.__setitem__(k, True))
    base.accept(f"{key}-up", lambda k=key: key_map.__setitem__(k, False))
fp_camera = base.render.attachNewNode("fp_camera")
fp_camera.setPos(base.cam.getPos())
fp_camera.setHpr(0, 0, 0)
base.cam.reparentTo(fp_camera)
base.cam.setPos(0, 0, 0)
sensitivity = 0.2
move_speed = 10
def update_camera(task):
    global heading, pitch
    dt = globalClock.getDt()
    if base.mouseWatcherNode.hasMouse():
        md = base.win.getPointer(0)
        x = md.getX()
        y = md.getY()
        center_x = base.win.getXSize() // 2
        center_y = base.win.getYSize() // 2
        dx = x - center_x
        dy = y - center_y
        base.win.movePointer(0, center_x, center_y)
        h = fp_camera.getH() - dx * sensitivity
        p = fp_camera.getP() - dy * sensitivity
        p = max(-89, min(89, p))  # Clamp pitch
        fp_camera.setH(h)
        fp_camera.setP(p)
    direction = Vec3(0, 0, 0)
    if key_map["w"]:
        direction += Vec3(0, 1, 0)
    if key_map["s"]:
        direction += Vec3(0, -1, 0)
    if key_map["a"]:
        direction += Vec3(-1, 0, 0)
    if key_map["d"]:
        direction += Vec3(1, 0, 0)
    if direction.lengthSquared() > 0:
        direction.normalize()
        movement = fp_camera.getQuat().xform(direction)
        movement.setZ(0)
        fp_camera.setPos(fp_camera.getPos() + movement * move_speed * dt)
    return task.cont
props = WindowProperties()
props.setCursorHidden(True)
props.setMouseMode(WindowProperties.M_relative)
base.win.requestProperties(props)
center_x = base.win.getXSize() // 2
center_y = base.win.getYSize() // 2
base.win.movePointer(0, center_x, center_y)
base.taskMgr.add(update_camera, "update_camera")
base.accept("escape", base.userExit)
base.run()