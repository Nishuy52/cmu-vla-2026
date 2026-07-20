import json
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-jason-cmu-ws/042376ea-8ed2-4b27-b155-0f949ec316c4/scratchpad")
import mirror_audit as M

out = {}
for scene in M.SCENES:
    try:
        out[scene] = M.audit_scene(scene)
        print(scene, "OK", out[scene]["n_blocked_total"], file=sys.stderr)
    except Exception as e:
        import traceback
        traceback.print_exc()
        out[scene] = {"error": str(e)}

json.dump(out, open("/tmp/claude-1000/-home-jason-cmu-ws/042376ea-8ed2-4b27-b155-0f949ec316c4/scratchpad/all_scenes.json", "w"), indent=2, default=str)
print("done")
