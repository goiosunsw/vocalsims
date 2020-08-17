import sys
from reed_up_downstream_dyn import ReedSimulation
from json_object import JSONObject



json_file = sys.argv[1]

with open(json_file) as f:
    js = JSONObject(f)


sim = ReedSimulation()
sim.from_json(js)
sim.simulate()

