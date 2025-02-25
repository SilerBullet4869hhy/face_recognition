import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import json

from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

import backend.face_detection as face_detection
from backend.face_match import compare_faces_api

app = FastAPI()

# 静态文件支持
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置Jinja2模板
env = Environment(loader=FileSystemLoader("templates"))

@app.get("/", response_class=HTMLResponse)
async def home():
    template = env.get_template("index.html")
    return template.render()
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 连接成功")

    try:
        while True:
            try:
                data = await websocket.receive_text()
                image_data = base64.b64decode(data.split(",")[1])
                np_arr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # 处理帧并返回人脸检测坐标
                frame, face_boxes = face_detection.detect_faces(frame)

                _, buffer = cv2.imencode(".jpg", frame)
                encoded_image = base64.b64encode(buffer).decode("utf-8")

                #response_data = json.dumps({"image": f"data:image/jpeg;base64,{encoded_image}", "faces": face_boxes})
                #await websocket.send_text(response_data)
                await websocket.send_text(f"data:image/jpeg;base64,{encoded_image}")

            except WebSocketDisconnect:
                print("WebSocket 断开")
                break
            except Exception as e:
                print(f"WebSocket 错误: {e}")
                continue

    finally:
        await websocket.close()
        print("WebSocket 连接关闭")

@app.post("/compare_faces")
async def compare_faces(request: dict):
    try:
        image_data = base64.b64decode(request["image"].split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="无效的图像数据")

        # **你可以在这里实现人脸比对算法**
        #match_result = "data:image/jpeg;base64," + base64.b64encode(image_data).decode("utf-8")  # 临时返回原图
        #return JSONResponse(content={"match": match_result})
        match_result = compare_faces_api(frame)
        return JSONResponse(content=match_result)

    except Exception as e:
        print("比对错误:", e)
        raise HTTPException(status_code=500, detail="人脸比对失败")


#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8800)