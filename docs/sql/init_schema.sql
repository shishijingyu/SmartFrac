-- ============================================
-- SmartFrac 第二阶段数据库建表脚本
-- 数据库：192.168.1.115:3306/SmartFrac
-- 字符集：utf8mb4
-- ============================================

CREATE DATABASE IF NOT EXISTS SmartFrac
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_general_ci;

USE SmartFrac;

-- ============================================
-- 1. 系统用户表
-- ============================================
DROP TABLE IF EXISTS sys_user;
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

-- 初始管理员
INSERT INTO sys_user (username, password, nickname, role)
VALUES ('admin', 'e10adc3949ba59abbe56e057f20f883e', '管理员', 'admin');

-- ============================================
-- 2. 项目表
-- ============================================
DROP TABLE IF EXISTS biz_project;
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

-- ============================================
-- 3. 算例表
-- ============================================
DROP TABLE IF EXISTS biz_case;
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

-- ============================================
-- 4. 仿真任务表
-- ============================================
DROP TABLE IF EXISTS sim_task;
CREATE TABLE sim_task (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '任务ID',
    task_no         VARCHAR(64)  NOT NULL COMMENT '任务编号',
    project_id      BIGINT       COMMENT '项目ID',
    case_id         BIGINT       COMMENT '算例ID',
    task_type       VARCHAR(32)  NOT NULL COMMENT '任务类型：nnp_train/pinn_solve/multiscale/uncertainty_sampling',
    status          VARCHAR(32)  DEFAULT 'pending' COMMENT '状态：pending/running/completed/failed/cancelled',
    progress        INT          DEFAULT 0 COMMENT '进度百分比0-100',
    input_files     VARCHAR(512) COMMENT '输入文件MinIO路径（JSON数组）',
    output_dir      VARCHAR(512) COMMENT '输出结果MinIO目录',
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

-- ============================================
-- 5. AI模型表
-- ============================================
DROP TABLE IF EXISTS ai_model;
CREATE TABLE ai_model (
    id              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '模型ID',
    name            VARCHAR(128) NOT NULL COMMENT '模型名称',
    model_type      VARCHAR(32)  NOT NULL COMMENT '类型：nnp/pinn/cnn/rl',
    version         VARCHAR(32)  DEFAULT 'v1.0' COMMENT '版本号',
    file_path       VARCHAR(512) COMMENT '模型文件MinIO路径（.pt/.pth）',
    metrics_json    JSON         COMMENT '模型评估指标',
    description     TEXT         COMMENT '模型说明',
    is_default      TINYINT      DEFAULT 0 COMMENT '是否默认：0否 1是',
    create_by       BIGINT       COMMENT '创建人',
    create_time     DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    is_deleted      TINYINT      DEFAULT 0 COMMENT '逻辑删除',
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI模型表';

-- ============================================
-- 6. 测试用例表
-- ============================================
DROP TABLE IF EXISTS test_case;
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

-- ============================================
-- 7. 测试执行记录表
-- ============================================
DROP TABLE IF EXISTS test_record;
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
