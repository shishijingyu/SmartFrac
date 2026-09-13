package vip.xiaonuo.smartfrac.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * 测试执行记录实体
 */
@Data
@TableName("test_record")
public class TestRecord {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long testCaseId;

    private Long taskId;

    private String result;

    private String actualMetrics;

    private String deviation;

    private Long tester;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}
