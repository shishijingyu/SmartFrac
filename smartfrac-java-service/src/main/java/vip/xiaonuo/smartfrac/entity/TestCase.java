package vip.xiaonuo.smartfrac.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * 测试用例实体
 */
@Data
@TableName("test_case")
public class TestCase {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private Long caseId;

    private String expectedMetrics;

    private String description;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableLogic
    private Integer isDeleted;
}
