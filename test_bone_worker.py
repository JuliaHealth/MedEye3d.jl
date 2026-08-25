import socket
import json
import base64
import numpy as np

shape = (64, 64, 32)
lesion = np.zeros(shape, dtype=np.uint8)
lesion[32, 32, 16] = 1

bone = np.zeros(shape, dtype=np.uint8)
bone[30:35, 30:35, 10:20] = 1

lesion_b64 = base64.b64encode(lesion.tobytes()).decode('ascii')
bone_b64 = base64.b64encode(bone.tobytes()).decode('ascii')

req = {
    "command": "bone_subsegmentation",
    "shape": list(shape),
    "spacing": [1.0, 1.0, 1.0],
    "lesion_mask_b64": lesion_b64,
    "bone_mask_b64": bone_b64
}

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 5005))
s.sendall(json.dumps(req).encode('utf-8'))

resp_data = b""
while True:
    chunk = s.recv(65536)
    if not chunk:
        break
    resp_data += chunk
s.close()

resp = json.loads(resp_data.decode('utf-8'))
print(resp["status"])
if resp["status"] == "success":
    surf = np.frombuffer(base64.b64decode(resp["surf_mask_b64"]), dtype=np.uint8).reshape(shape)
    marr = np.frombuffer(base64.b64decode(resp["marr_mask_b64"]), dtype=np.uint8).reshape(shape)
    print("Surf sum:", surf.sum())
    print("Marr sum:", marr.sum())
else:
    print(resp.get("message"))
