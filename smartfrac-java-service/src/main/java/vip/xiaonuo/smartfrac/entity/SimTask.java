package vip.xiaonuo.smartfrac.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * 仿真任务实体
 */
@Data
@TableName("sim_task")
public class SimTask {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String taskNo;

    private Long projectId;

    private Long caseId;

    private String taskType;

    private String status;

    private Integer progress;

    private String inputFiles;

    private String outputDir;

    private String metricsJson;

    private String errorMsg;

    private Long pid;

    private LocalDateTime startTime;

    private LocalDateTime endTime;

    private Long createBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;

    @TableLogic
    private Integer isDeleted;
}
