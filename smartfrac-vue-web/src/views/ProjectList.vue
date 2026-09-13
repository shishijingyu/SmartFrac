<template>
  <div>
    <el-card shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>项目管理</span>
          <el-button type="primary" :icon="Plus" @click="openDialog">新建项目</el-button>
        </div>
      </template>
      <el-table :data="projects" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="name" label="项目名称" />
        <el-table-column prop="description" label="描述" show-overflow-tooltip />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'active' ? 'success' : 'info'" size="small">
              {{ row.status === 'active' ? '进行中' : '已归档' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="创建时间" width="180" />
        <el-table-column label="操作" width="180">
          <template #default="{ row }">
            <el-button size="small" @click="openDialog(row)">编辑</el-button>
            <el-button size="small" type="danger" @click="handleDelete(row.id)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 新建/编辑对话框 -->
    <el-dialog v-model="dialogVisible" title="项目信息" width="500px">
      <el-form :model="form" label-width="80px">
        <el-form-item label="项目名称">
          <el-input v-model="form.name" placeholder="请输入项目名称" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" type="textarea" rows="3" placeholder="请输入项目描述" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSave">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Plus } from '@element-plus/icons-vue'
import { projectApi } from '../api'
import { ElMessage } from 'element-plus'

const projects = ref([])
const loading = ref(false)
const dialogVisible = ref(false)
const form = ref({ id: null, name: '', description: '' })

const loadProjects = async () => {
  loading.value = true
  try {
    const res = await projectApi.list()
    projects.value = res.data || []
  } catch (e) {
    ElMessage.error('加载项目列表失败')
  } finally {
    loading.value = false
  }
}

const openDialog = (row) => {
  form.value = row ? { ...row } : { id: null, name: '', description: '' }
  dialogVisible.value = true
}

const handleSave = async () => {
  try {
    if (form.value.id) {
      await projectApi.update(form.value.id, form.value)
    } else {
      await projectApi.create(form.value)
    }
    ElMessage.success('保存成功')
    dialogVisible.value = false
    loadProjects()
  } catch (e) {
    ElMessage.error('保存失败')
  }
}

const handleDelete = async (id) => {
  try {
    await projectApi.delete(id)
    ElMessage.success('删除成功')
    loadProjects()
  } catch (e) {
    ElMessage.error('删除失败')
  }
}

onMounted(loadProjects)
</script>
