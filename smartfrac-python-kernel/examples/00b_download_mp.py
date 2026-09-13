"""从 Materials Project 下载 α-Fe DFT 数据集（需要 API key）。

申请步骤（免费，1 分钟）：
1. 打开 https://next-gen.materialsproject.org/
2. 注册 / 登录（Google 账号即可）
3. 点右上角头像 -> "API Molecules" -> "API KEYS" -> 复制 key
4. 把 key 填到下面 MP_API_KEY 变量，或设环境变量：
   PowerShell:  $env:MP_API_KEY="你的key"
5. 运行: python examples/00b_download_mp.py

依赖: pip install mp-api
"""
from __future__ import annotations

import os
from pathlib import Path

from ase.io import write

MP_API_KEY = os.environ.get("MP_API_KEY", "在此粘贴你的 key")
OUT = Path("data/raw/alpha_fe_mp")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if "在此粘贴" in MP_API_KEY:
        print("请先设置 MP_API_KEY 环境变量，或在脚本里填入你的 key。")
        print("申请地址: https://next-gen.materialsproject.org/apikeys")
        return

    try:
        from mp_api.client import MPRester
    except ImportError:
        print("请先: pip install mp-api")
        return

    frames = []
    with MPRester(MP_API_KEY) as mpr:
        # 查询所有含 Fe 的结构
        docs = mpr.materials.search(elements=["Fe"], num_elements=1,
                                    fields=["structure", "energy", "forces"])
        print(f"MP 查到 {len(docs)} 条 Fe 记录")
        for doc in docs[:500]:  # 先下 500 个够 MVP 用
            at = doc.structure.to_ase_atoms()
            # MP 不直接给 forces 数组，需要自己跑；这里先收结构
            frames.append(at)

    out_file = OUT / "alpha_fe_mp.xyz"
    write(str(out_file), frames)
    print(f"已保存 {len(frames)} 个构型 -> {out_file}")
    print("下一步: 用 VASP 对这些构型做单点计算，把能量/力标签写进 xyz。")


if __name__ == "__main__":
    main()
