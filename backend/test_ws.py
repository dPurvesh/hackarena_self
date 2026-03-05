import asyncio, websockets, json

async def t():
    async with websockets.connect("ws://localhost:8000/ws/live") as ws:
        for i in range(3):
            msg = await ws.recv()
            d = json.loads(msg)
            cams = list(d.get("cameras", {}).keys())
            top_has_frame = "frame" in d
            cam_has_frame = any(c.get("frame") for c in d.get("cameras", {}).values())
            score = d.get("score", 0)
            msg_kb = len(msg) / 1024
            print(f"msg {i+1}: cams={cams} top_frame={top_has_frame} cam_frame={cam_has_frame} score={score} size={msg_kb:.1f}KB")
        print("WS OK")

asyncio.run(t())
