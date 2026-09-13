package vip.xiaonuo.smartfrac.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import vip.xiaonuo.smartfrac.common.Result;
import vip.xiaonuo.smartfrac.entity.BizCase;
import vip.xiaonuo.smartfrac.service.BizCaseService;

import java.util.List;

/**
 * 算例管理 Controller
 */
@RestController
@RequestMapping("/api/cases")
public class CaseController {

    @Autowired
    private BizCaseService caseService;

    @GetMapping
    public Result<List<BizCase>> list(@RequestParam(required = false) Long projectId) {
        LambdaQueryWrapper<BizCase> wrapper = new LambdaQueryWrapper<>();
        if (projectId != null) {
            wrapper.eq(BizCase::getProjectId, projectId);
        }
        wrapper.orderByDesc(BizCase::getCreateTime);
        return Result.success(caseService.list(wrapper));
    }

    @GetMapping("/{id}")
    public Result<BizCase> getById(@PathVariable Long id) {
        return Result.success(caseService.getById(id));
    }

    @PostMapping
    public Result<BizCase> create(@RequestBody BizCase bizCase) {
        caseService.save(bizCase);
        return Result.success(bizCase);
    }

    @PutMapping("/{id}")
    public Result<BizCase> update(@PathVariable Long id, @RequestBody BizCase bizCase) {
        bizCase.setId(id);
        caseService.updateById(bizCase);
        return Result.success(bizCase);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        caseService.removeById(id);
        return Result.success();
    }
}
