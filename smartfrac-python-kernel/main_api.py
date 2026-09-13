"""SmartFrac Python 计算内核 FastAPI 服务入口。

第二阶段新增：将原有算法内核包装为 HTTP 服务，供 Java SpringBoot 调用。
端口：48092（Java 后端 48091，前端 91）

运行方式：
    cd smartfrac-python-kernel
    pip install fastapi uvicorn
    uvicorn main_api:app --host 0.0.0.0 --port 48092
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SmartFrac Python Kernel", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    config_path: str = "configs/nnp_alpha_fe.yaml"
    epochs: int | None = None


class MultiscaleRequest(BaseModel):
    case_id: int
    case_type: str = "multiscale"
    material: str = "alpha_Fe"
    grain_size: float = 10.0  # 晶粒尺寸 μm
    crack_length: float = 5.0  # 裂纹半长 mm
    load_stress: float = 200.0  # 远场应力 MPa


class UncertaintyRequest(BaseModel):
    case_id: int
    n_samples: int = 100  # 采样样本数
    distribution: str = "normal"  # normal/lognormal/uniform
    param: str = "load"  # 扰动参数：load/grain_orientation/defect_position


@app.get("/api/health")
def health():
    """健康检查"""
    return {
        "status": "UP",
        "service": "smartfrac-python-kernel",
        "version": "2.0.0",
        "modules": ["nnp", "pinn", "md", "io_utils", "viz", "multiscale", "uncertainty"],
    }


@app.post("/api/nnp/train")
async def train_nnp(req: TrainRequest):
    """提交 NNP 训练任务（占位接口，后续接入 RabbitMQ 异步队列）"""
    return {
        "task_id": f"nnp_{int(time.time())}",
        "status": "accepted",
        "message": "NNP 训练任务已接收，待接入 RabbitMQ 异步执行",
        "config": req.config_path,
    }


@app.post("/api/pinn/solve")
async def solve_pinn(req: TrainRequest):
    """提交 PINN 求解任务"""
    return {
        "task_id": f"pinn_{int(time.time())}",
        "status": "accepted",
        "message": "PINN 求解任务已接收，待接入 RabbitMQ 异步执行",
    }


@app.post("/api/multiscale/run")
async def run_multiscale(req: MultiscaleRequest):
    """提交多尺度耦合任务（M5模块）"""
    return {
        "task_id": f"multiscale_{int(time.time())}",
        "status": "accepted",
        "message": "多尺度耦合任务已接收，待实现层级迁移算法",
        "params": req.dict(),
    }


@app.post("/api/uncertainty/run")
async def run_uncertainty(req: UncertaintyRequest):
    """提交不确定性量化任务（M6模块）"""
    return {
        "task_id": f"uncertainty_{int(time.time())}",
        "status": "accepted",
        "message": f"不确定性量化任务已接收，采样数={req.n_samples}",
        "params": req.dict(),
    }


@app.get("/api/models/list")
def list_models():
    """列出已有模型 checkpoint"""
    results_dir = ROOT / "results"
    models = []
    if results_dir.exists():
        for pt in results_dir.rglob("*.pt"):
            models.append({
                "name": pt.stem,
                "path": str(pt.relative_to(ROOT)),
                "size_kb": round(pt.stat().st_size / 1024, 1),
            })
    return {"models": models, "total": len(models)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=48092)
