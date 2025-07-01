from panda3d.core import Filename, DSearchPath

filename = Filename("models/environment.egg.pz")

search = DSearchPath()
search.append_directory(".")
search.append_directory("models")

if filename.resolve_filename(search):
    print("✅ Resolved path:", filename)
else:
    print("❌ Could not resolve file.")
