# SmartFrac 第二阶段（V2.0β）开发设计说明书

> 依据：《多尺度-不确定性耦合AI断裂智能仿真系统_第二阶段需求V1.0》
> 技术栈：Java SpringBoot 3 + Vue3 + ElementPlus + Python FastAPI + MySQL 8 + Redis
> 数据库：192.168.1.115:3306/SmartFrac  root/root1234
> Redis：192.168.1.115:6379  无密码

---

## 一、系统总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  L7  用户交互层（Vue3 + ElementPlus + vtk.js + ECharts）        │
│  端口: 91                                                       │
│  算例配置 / 任务队列 / 结果可视化 / 报告导出 / 回归测试          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│  Java SpringBoot 业务服务层（端口: 48091）                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ 项目管理  │ │ 任务调度  │ │ 模型管理  │ │ 测试套件  │            │
│  └──────────┘ └────┬─────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌────▼─────┐ ┌──────────┐ ┌──────────┐            │
│  │ 用户权限  │ │ 报告生成  │ │ 审计日志  │ │ 文件管理  │            │
│  └──────────┘ └──────────┘ └──────────┘ └────┬─────┘            │
└────────────────────────────────────────────────┼───────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          │                      │                      │
                   ┌──────▼──────┐        ┌──────▼──────┐
                   │  MySQL 8.0   │        │    Redis     │
                   │  192.168..  │        │  192.168..  │
                   │  :3306       │        │  :6379       │
                   └───────────────┘        └──────────────┘
                                                                          │
┌──────────────────────────────────────────────────────────────────────────▼────────┐
│  Python 计算内核服务层（FastAPI，端口: 48092）                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ M5 多尺度    │ │ M6 不确定性  │ │ M2-A NNP    │ │ M2-B PINN   │               │
│  │ 耦合内核     │ │ 量化引擎    │ │ 原子势      │ │ 连续介质    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                              │
│  │ CNN 失效识别 │ │ RL 断裂判据 │ │ MD 加速     │                              │
│  └─────────────┘ └─────────────┘ └─────────────┘                              │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Java 后端模块设计

### 2.1 包结构

```
vip.xiaonuo.smartfrac/
├── SmartFracApplication.java          # 启动类
├── common/                            # 公共组件
│   ├── Result.java                     # 统一返回结果
│   ├── PageResult.java                 # 分页返回
│   ├── exception/                      # 全局异常处理
│   └── utils/                          # 工具类
├── config/                            # 配置类
│   ├── CorsConfig.java                 # 跨域
│   ├── MyBatisPlusConfig.java          # MyBatis-Plus分页插件
│   ├── RedisConfig.java               # Redis缓存/任务队列配置
│   └── WebSocketConfig.java            # WebSocket进度推送
├── controller/                        # 接口层
│   ├── ProjectController.java          # 项目管理
│   ├── CaseController.java             # 算例管理
│   ├── TaskController.java             # 仿真任务
│   ├── ModelController.java            # 模型管理
│   ├── TestSuiteController.java        # 测试套件
│   ├── ReportController.java           # 报告导出
│   └── HealthController.java           # 健康检查
├── service/                           # 业务层
│   ├── ProjectService.java
│   ├── CaseService.java
│   ├── TaskService.java
│   ├── ModelService.java
│   ├── TestSuiteService.java
│   └── impl/
├── mapper/                            # 数据访问层
├── entity/                            # 数据库实体
├── dto/                               # 请求/响应DTO
├── redis/                            # Redis任务队列
│   ├── TaskQueueService.java          # 任务生产者/消费者
│   └── TaskProgressService.java       # 任务进度缓存
└── websocket/                         # WebSocket进度推送
```

### 2.2 核心业务流程

**提交仿真任务：**
1. Vue 前端选择算例 → POST `/api/tasks`
2. Java 校验参数 → 写入 `sim_task` 表（状态：排队中）
3. Java 推入 Redis 任务队列（任务ID + yaml配置 + 输入文件路径）
4. Python 从 Redis BRPOP 消费任务 → 执行仿真 → WebSocket 回传进度
5. 仿真完成 → Python 更新任务状态到 Redis → 通知 Java
6. Java 更新任务状态为已完成 → 前端轮询/WebSocket收到通知

---

## 三、数据库设计（MySQL 8.0）

数据库：`192.168.1.115:3306/SmartFrac`

### 3.1 用户与项目表

```sql
-- 用户表
CREATE TABLE sys_user (
    id          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '用户ID',
    username    VARCHAR(64)  NOT NULL COMMENT '用户名',
    password    VARCHAR(128) NOT NULL COMMENT '密码（加密存储）',
    nickname    VARCHAR(64)  DEFAULT NULL COMMENT '昵称',
    role        VARCHAR(32)  DEFAULT 'user' COMMENT '角色：admin/user/tester',
    status      TINYINT      DEFAULT 1 COMMENT '状态：0禁用 1正常',
    create_time DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    is_deleted  TINYINT      DEFAULT 0 COMMENT '逻辑删除：0未删 1已删',
    PRIMARY KEY (id),
    UNIQUE KEY uk_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='系统用户表';

-- 项目表
CREATE TABLE biz_project (
    id          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '项目ID',
    name        VARCHAR(128) NOT NULL COMMENT '项目名称',
    description TEXT         COMMENT '项目描述',
    owner_id    BIGINT       COMMENT '负责人ID',
    status      VARCHAR(32)  DEFAULT 'active' COMMENT '状态：active/archived',
    create_time DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    is_deleted  TINYINT      DEFAULT 0 COMMENT '逻辑删除',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='项目表';
```

### 3.2 算例与配置表

```sql
-- 算例表
CREATE TABLE biz_case (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '算例ID',
    project_id      BIGINT       COMMENT '所属项目ID',
    name            VARCHAR(128) NOT NULL COMMENT '算例名称',
    case_type       VARCHAR(32)  COMMENT '算例类型：nnp/pinn/multiscale/uncertainty',
    material        VARCHAR(64)  DEFAULT 'alpha_Fe' COMMENT '材料：alpha_Fe等',
    yaml_config     LONGTEXT     COMMENT '算例YAML配置内容',
    description     TEXT         COMMENT '算例说明',
    create_by       BIGINT       COMMENT '创建人',
    create_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time     DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    is_deleted      TINYINT      DEFAULT 0 COMMENT '逻辑删除',
    PRIMARY KEY (id),
    KEY idx_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='算例表';
```

### 3.3 仿真任务表

```sql
-- 仿真任务表
CREATE TABLE sim_task (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '任务ID',
    task_no         VARCHAR(64)  NOT NULL COMMENT '任务编号',
    project_id      BIGINT       COMMENT '项目ID',
    case_id         BIGINT       COMMENT '算例ID',
    task_type       VARCHAR(32)  NOT NULL COMMENT '任务类型：nnp_train/pinn_solve/multiscale/uncertainty_sampling',
    status          VARCHAR(32)  DEFAULT 'pending' COMMENT '状态：pending/running/completed/failed/cancelled',
    progress        INT          DEFAULT 0 COMMENT '进度百分比0-100',
    input_files     VARCHAR(512) COMMENT '输入文件路径（JSON数组）',
    output_dir      VARCHAR(512) COMMENT '输出结果目录',
    metrics_json    JSON         COMMENT '结果指标（E-MAE/F-RMSE/L2/加速比等）',
    error_msg       TEXT         COMMENT '错误信息',
    pid             BIGINT       COMMENT 'Python进程ID',
    start_time      DATETIME     COMMENT '开始时间',
    end_time        DATETIME     COMMENT '结束时间',
    create_by       BIGINT       COMMENT '提交人',
    create_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time     DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    is_deleted      TINYINT      DEFAULT 0 COMMENT '逻辑删除',
    PRIMARY KEY (id),
    UNIQUE KEY uk_task_no (task_no),
    KEY idx_status (status),
    KEY idx_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='仿真任务表';
```

### 3.4 模型管理表

```sql
-- AI模型表
CREATE TABLE ai_model (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '模型ID',
    name            VARCHAR(128) NOT NULL COMMENT '模型名称',
    model_type      VARCHAR(32)  NOT NULL COMMENT '类型：nnp/pinn/cnn/rl',
    version         VARCHAR(32)  DEFAULT 'v1.0' COMMENT '版本号',
    file_path       VARCHAR(512) COMMENT '模型文件路径（.pt/.pth）',
    metrics_json    JSON         COMMENT '模型评估指标',
    description     TEXT         COMMENT '模型说明',
    is_default      TINYINT      DEFAULT 0 COMMENT '是否默认：0否 1是',
    create_by       BIGINT       COMMENT '创建人',
    create_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    is_deleted      TINYINT      DEFAULT 0 COMMENT '逻辑删除',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI模型表';
```

### 3.5 测试用例与回归表

```sql
-- 测试用例表
CREATE TABLE test_case (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '用例ID',
    name            VARCHAR(128) NOT NULL COMMENT '用例名称',
    case_id         BIGINT       COMMENT '关联算例ID',
    expected_metrics JSON        COMMENT '预期指标阈值',
    description     TEXT         COMMENT '用例说明',
    create_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    is_deleted      TINYINT      DEFAULT 0 COMMENT '逻辑删除',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试用例表';

-- 测试执行记录表
CREATE TABLE test_record (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '记录ID',
    test_case_id    BIGINT       COMMENT '测试用例ID',
    task_id         BIGINT       COMMENT '对应仿真任务ID',
    result          VARCHAR(32)  COMMENT '结果：pass/fail',
    actual_metrics  JSON         COMMENT '实际指标',
    deviation       TEXT         COMMENT '偏差说明',
    tester          BIGINT       COMMENT '测试人',
    create_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_test_case (test_case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='测试执行记录表';
```

---

## 四、REST API 接口设计

### 4.1 接口规范

- Base URL：`http://localhost:48091/api`
- 认证：后续阶段加入JWT，第二阶段先不做严格权限
- 返回格式：`{ code: 200, message: "success", data: {...} }`

### 4.2 接口清单

| 模块 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 健康检查 | GET | /api/health | 服务状态 |
| **项目** | GET | /api/projects | 项目列表 |
| | POST | /api/projects | 创建项目 |
| | PUT | /api/projects/{id} | 更新项目 |
| | DELETE | /api/projects/{id} | 删除项目 |
| **算例** | GET | /api/cases | 算例列表 |
| | GET | /api/cases/{id} | 算例详情 |
| | POST | /api/cases | 创建算例 |
| | PUT | /api/cases/{id} | 更新算例 |
| | DELETE | /api/cases/{id} | 删除算例 |
| | POST | /api/cases/{id}/upload-yaml | 上传YAML配置 |
| **任务** | GET | /api/tasks | 任务列表（支持状态筛选） |
| | GET | /api/tasks/{id} | 任务详情 |
| | POST | /api/tasks | 提交仿真任务 |
| | POST | /api/tasks/{id}/cancel | 取消任务 |
| | GET | /api/tasks/{id}/progress | 查询进度（WebSocket为主） |
| **模型** | GET | /api/models | 模型列表 |
| | GET | /api/models/{id} | 模型详情 |
| | POST | /api/models/upload | 上传模型文件 |
| | PUT | /api/models/{id}/set-default | 设为默认模型 |
| **测试** | GET | /api/test-cases | 测试用例列表 |
| | POST | /api/test-cases | 新建测试用例 |
| | POST | /api/test-suite/run | 执行回归测试套件 |
| | GET | /api/test-records | 测试执行记录 |
| **报告** | GET | /api/reports/{taskId}/pdf | 导出PDF报告 |
| | GET | /api/reports/{taskId}/word | 导出Word报告 |

---

## 五、Vue3 前端页面设计

### 5.1 页面路由

```
/login              # 登录页（第二阶段后期加）
/                   # 主布局
├── /dashboard      # 首页概览（统计卡片+系统状态）
├── /projects       # 项目管理
├── /cases          # 算例管理
│   └── /cases/:id/config  # 算例参数配置（YAML可视化表单）
├── /tasks          # 任务队列
│   └── /tasks/:id        # 任务详情（进度+日志+结果预览）
├── /models         # 模型管理
├── /testing        # 验证测试
│   ├── /testing/cases     # 测试用例库
│   └── /testing/records   # 回归测试报告
└── /reports        # 报告中心
```

### 5.2 核心页面组件

| 页面 | 核心组件 |
|------|---------|
| 算例配置 | 动态表单（从YAML schema生成）+ YAML编辑器 |
| 任务队列 | 任务状态表格 + 实时进度条（WebSocket）+ 日志查看 |
| 结果可视化 | vtk.js三维场渲染 + ECharts统计图表 + 指标面板 |
| 回归测试 | 测试用例勾选 + 批量执行 + 结果对比表格 |

---

## 六、Python 内核扩展设计

### 6.1 新增模块

在 `smartfrac-python-kernel/src/smartfrac/` 下新增：

```
├── multiscale/              # M5 多尺度耦合内核（新增）
│   ├── __init__.py
│   ├── transfer.py          # 层级知识迁移接口（接口E）
│   ├── hybrid_solver.py     # AI代理-传统求解混合（接口F）
│   ├── consistency_check.py # 物理一致性校验
│   └── fidelity_budget.py  # 保真度-算力调节
├── uncertainty/             # M6 不确定性量化引擎（新增）
│   ├── __init__.py
│   ├── sampler.py           # 随机采样（正态/对数正态/均匀）
│   ├── cnn_classifier.py    # CNN失效模式识别
│   ├── rl_criterion.py       # RL统计断裂判据
│   └── propagation.py       # 不确定性传播与概率输出
└── api/                     # FastAPI接口层（新增）
    ├── __init__.py
    ├── task_consumer.py     # Redis任务消费（BRPOP）
    ├── progress_ws.py       # WebSocket进度推送
    └── redis_client.py       # Redis客户端（任务队列/进度缓存）
```

### 6.2 FastAPI 接口

```
POST /api/nnp/train          # 提交NNP训练任务
POST /api/pinn/solve          # 提交PINN求解任务
POST /api/multiscale/run      # 提交多尺度耦合任务
POST /api/uncertainty/run     # 提交不确定性量化任务
GET  /api/tasks/{task_id}/status  # 查询任务状态
GET  /api/models/list         # 列出可用模型
```

---

## 七、中间件配置

### 7.1 Redis 任务队列设计

```
地址：192.168.1.115:6379  无密码

任务队列（List结构）：
  smartfrac:queue:nnp.train        # NNP训练任务
  smartfrac:queue:pinn.solve       # PINN求解任务
  smartfrac:queue:multiscale.run   # 多尺度耦合任务
  smartfrac:queue:uncertainty.run  # 不确定性量化任务

任务详情（String结构，过期1小时）：
  smartfrac:task:{taskId}          # 任务参数JSON

任务进度（String结构，过期30分钟）：
  smartfrac:task:{taskId}:progress  # status:progress
```

### 7.2 文件存储

结果文件直接存储在 Python 内核服务器本地磁盘，通过 HTTP 接口下载，不引入独立对象存储。

---

## 八、开发里程碑（12个月）

| 阶段 | 时间 | 交付内容 |
|------|------|---------|
| S1 | 第1月 | 数据库建表 + Java项目骨架 + Vue框架搭建 + FastAPI骨架 |
| S2 | 第2-3月 | 项目/算例CRUD + 任务提交 + Redis任务队列打通 + Python内核API |
| S3 | 第4-5月 | M5多尺度耦合内核 + 层级迁移接口 + 物理校验 |
| S4 | 第6-7月 | M6不确定性量化引擎 + 随机采样 + 概率输出 |
| S5 | 第8-9月 | CNN失效识别 + RL断裂判据 + 可视化概率云图 |
| S6 | 第10月 | 批量任务调度 + 测试套件 + 回归测试 |
| S7 | 第11月 | 报告导出 + 全量集成测试 + Bug修复 |
| S8 | 第12月 | V2.0β验收 + 内部测试 + 文档完善 |

---

## 九、第二阶段不做的边界

- ❌ 完整用户权限系统（第二阶段先简单登录，不做细粒度RBAC）
- ❌ 云端部署 / Kubernetes
- ❌ 疲劳断裂、氢脆等复杂物理模型
- ❌ 分布式多节点并行
- ❌ 闭环自进化学习
- ❌ 完整的商业级安全加固

---

> 本文档为第二阶段开发设计基线，后续开发以本文档为准。
