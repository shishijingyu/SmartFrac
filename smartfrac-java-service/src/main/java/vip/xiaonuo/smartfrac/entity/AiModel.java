package vip.xiaonuo.smartfrac.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * AI模型实体
 */
@Data
@TableName("ai_model")
public class AiModel {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String modelType;

    private String version;

    private String filePath;

    private String metricsJson;

    private String description;

    private Integer isDefault;

    private Long createBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableLogic
    private Integer isDeleted;
}
