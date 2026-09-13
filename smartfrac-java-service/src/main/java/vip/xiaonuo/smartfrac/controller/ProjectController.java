package vip.xiaonuo.smartfrac.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import vip.xiaonuo.smartfrac.common.Result;
import vip.xiaonuo.smartfrac.entity.BizProject;
import vip.xiaonuo.smartfrac.service.BizProjectService;

import java.util.List;

/**
 * 项目管理 Controller
 */
@RestController
@RequestMapping("/api/projects")
public class ProjectController {

    @Autowired
    private BizProjectService projectService;

    @GetMapping
    public Result<List<BizProject>> list() {
        return Result.success(projectService.list());
    }

    @GetMapping("/{id}")
    public Result<BizProject> getById(@PathVariable Long id) {
        return Result.success(projectService.getById(id));
    }

    @PostMapping
    public Result<BizProject> create(@RequestBody BizProject project) {
        projectService.save(project);
        return Result.success(project);
    }

    @PutMapping("/{id}")
    public Result<BizProject> update(@PathVariable Long id, @RequestBody BizProject project) {
        project.setId(id);
        projectService.updateById(project);
        return Result.success(project);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        projectService.removeById(id);
        return Result.success();
    }
}
