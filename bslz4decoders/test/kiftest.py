


from silx.resources import ExternalResources
filename = ExternalResources("silx", "http://www.silx.org/pub/silx").getfile("hplc.h5")

print(filename)
import os, sys
cmd = f"python3 bench_parsechunks.py {filename} /entry_0000/measurement/data"
print(cmd)

print(os.popen(cmd).read())
